"""验收一：AscendGCNConv / ascend_global_mean_pool 与 PyG 原版算子数值等价。

两种输入：
  1) 正常图：多节点（env/tr/cs/ev 四类）多边；
  2) reset 极端边界图：仿真 reset 第 0 步的真实状态（1 个 env 节点、0 条边）。

每项都比较
  - 前向输出最大绝对误差
  - 反向梯度最大绝对误差（权重 / bias / 输入特征）
要求全部 < 1e-5。

运行： .venv\\Scripts\\python.exe test_ascend_equiv.py
"""
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
# 兼容两种摆放：脚本在仓库外（BASE/EV-GNN）或直接放在仓库根目录
EVGNN = BASE / "EV-GNN" if (BASE / "EV-GNN").is_dir() else BASE
sys.path.insert(0, str(EVGNN))
os.chdir(EVGNN)

import numpy as np
import torch
from torch_geometric.nn import GCNConv, global_mean_pool

from utils.ascend_layers import (AscendGCNConv, ascend_global_mean_pool,
                                 ascend_gcn_norm)
from utils.device import get_device

TOL = 1e-5
NPU_TOL = 1e-4        # 昇腾上 float32 的访存/指令差异，放宽到 1e-4
results = []


def _record(name, fwd_err, grad_errs, extra="", tol=None):
    tol = TOL if tol is None else tol
    grad_max = max(grad_errs) if grad_errs else 0.0
    ok = fwd_err < tol and grad_max < tol
    results.append((name, fwd_err, grad_max, ok, extra))
    print(f"  {name:<34s} fwd={fwd_err:.3e}  grad={grad_max:.3e}  "
          f"{'PASS' if ok else 'FAIL'} {extra}")


def compare_conv(x, edge_index, tag, in_dim=8, out_dim=16, seed=0):
    torch.manual_seed(seed)
    pyg = GCNConv(in_dim, out_dim)
    asc = AscendGCNConv(in_dim, out_dim)   # 参数名与 PyG 一致，可直接 load
    asc.load_state_dict(pyg.state_dict(), strict=True)

    w = torch.randn(x.size(0), out_dim)

    x1 = x.clone().detach().requires_grad_(True)
    out1 = pyg(x1, edge_index)
    (out1 * w).sum().backward()

    x2 = x.clone().detach().requires_grad_(True)
    out2 = asc(x2, edge_index)
    (out2 * w).sum().backward()

    fwd = (out1 - out2).abs().max().item()
    grads = [
        (pyg.lin.weight.grad - asc.lin.weight.grad).abs().max().item(),
        (pyg.bias.grad - asc.bias.grad).abs().max().item(),
        (x1.grad - x2.grad).abs().max().item(),
    ]
    _record(f"AscendGCNConv[{tag}]", fwd, grads,
            f"形状={tuple(out1.shape)}")


def compare_pool(x, batch, tag):
    w = torch.randn(*(int(batch.max().item()) + 1, x.size(1)))

    x1 = x.clone().detach().requires_grad_(True)
    out1 = global_mean_pool(x1, batch)
    (out1 * w).sum().backward()

    x2 = x.clone().detach().requires_grad_(True)
    out2 = ascend_global_mean_pool(x2, batch)
    (out2 * w).sum().backward()

    fwd = (out1 - out2).abs().max().item()
    grads = [(x1.grad - x2.grad).abs().max().item()]
    _record(f"ascend_global_mean_pool[{tag}]", fwd, grads,
            f"输出={tuple(out1.shape)}")


def normal_graph():
    """17 节点（1 env + 2 tr + 5 cs + 9 ev），双向边。"""
    env = [0]
    trs = [1, 2]
    cs = [3, 4, 5, 6, 7]
    evs = list(range(8, 17))

    pairs = []
    for t in trs:
        pairs.append((env[0], t))
    for c, t in zip(cs, [1, 1, 2, 2, 2]):
        pairs.append((t, c))
    for i, e in enumerate(evs):
        pairs.append((cs[i % len(cs)], e))

    src, dst = [], []
    for a, b in pairs:
        src += [a, b]
        dst += [b, a]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return torch.randn(17, 8), edge_index


def reset_graph():
    """真实仿真 reset 第 0 步的图（1 节点 0 边）；失败则退回合成图。"""
    try:
        import gymnasium as gym
        import yaml
        from ev2gym.rl_agent.reward import SimpleReward
        from utils.state import PublicPST_GNN

        cfg = str(EVGNN / "config_files" / "PublicPST.yaml")
        yaml.safe_load(open(cfg, "r", encoding="utf-8"))
        gym.envs.register(id="evs-equiv-v1",
                          entry_point="ev2gym.models.ev2gym_env:EV2Gym",
                          kwargs={"config_file": cfg,
                                  "generate_rnd_game": True,
                                  "reward_function": SimpleReward,
                                  "state_function": PublicPST_GNN},
                          )
        env = gym.make("evs-equiv-v1")
        state, _ = env.reset()
        edge_index = torch.from_numpy(np.asarray(state.edge_index)).long()
        x = torch.randn(int(state.node_types.shape[0]), 8)
        extra = (f"来源=真实 reset（节点 {x.size(0)}、边 {edge_index.size(1)}、"
                 f"ev_indexes 长度 {len(state.ev_indexes)}）")
        return x, edge_index, extra
    except Exception as exc:
        print(f"  [warn] 真实 reset 不可用（{exc}），改用合成边界图")
        x = torch.randn(1, 8)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        return x, edge_index, "来源=合成（1 节点 0 边）"


def dense_reference_check(src, dst, num_nodes=6, in_dim=8, out_dim=16):
    """用稠密矩阵 D^-1/2 (A+I) D^-1/2 X W + b 独立复核（不经过 PyG）。"""
    torch.manual_seed(7)
    conv = AscendGCNConv(in_dim, out_dim)
    x = torch.randn(num_nodes, in_dim)
    out = conv(x, torch.tensor([src, dst], dtype=torch.long))

    A = np.zeros((num_nodes, num_nodes))
    for s, d in zip(src, dst):   # PyG 语义：A[target, source]
        A[d, s] = 1.0
    A += np.eye(num_nodes)
    deg = A.sum(1)
    Dn = np.diag(deg ** -0.5)
    W = conv.lin.weight.detach().numpy()
    ref = Dn @ A @ Dn @ (x.numpy() @ W.T) + conv.bias.detach().numpy()
    _record("稠密公式复核[AscendGCNConv]", float(np.abs(out.detach().numpy() - ref).max()),
            [0.0], "独立于 PyG")

    xg = x.clone().requires_grad_(True)
    conv(xg, torch.tensor([src, dst], dtype=torch.long)).sum().backward()
    assert float(xg.grad.abs().sum()) > 0, "梯度为 0，说明前向不可导"


def npu_check():
    """实机 NPU：把算子与两张图搬到 npu:0，前向+反向与 CPU 结果比对。"""
    dev = get_device("auto")
    if dev.type != "npu":
        print(f"\n[5] NPU 实机校验：当前设备 {dev}，无 NPU，跳过"
              "（CPU 7 项已通过即可上机）")
        return

    print(f"\n[5] NPU 实机校验（{dev}，容差 {NPU_TOL:g}）")
    x_norm, ei_norm = normal_graph()
    x_bd, ei_bd, _ = reset_graph()

    for tag, x, ei in (("正常图17节点", x_norm, ei_norm),
                       ("边界图1节点0边", x_bd, ei_bd)):
        torch.manual_seed(11)
        cpu_conv = AscendGCNConv(8, 16)
        npu_conv = AscendGCNConv(8, 16)
        npu_conv.load_state_dict(cpu_conv.state_dict())
        npu_conv = npu_conv.to(dev)

        w = torch.randn(x.size(0), 16)

        xc = x.clone().detach().requires_grad_(True)
        out_c = cpu_conv(xc, ei)
        (out_c * w).sum().backward()

        xn = x.clone().detach().to(dev).requires_grad_(True)
        out_n = npu_conv(xn, ei.to(dev))
        (out_n * w.to(dev)).sum().backward()

        fwd = (out_n.detach().cpu() - out_c.detach()).abs().max().item()
        grads = [
            (npu_conv.lin.weight.grad.cpu() - cpu_conv.lin.weight.grad).abs().max().item(),
            (npu_conv.bias.grad.cpu() - cpu_conv.bias.grad).abs().max().item(),
            (xn.grad.cpu() - xc.grad).abs().max().item(),
        ]
        _record(f"NPU AscendGCNConv[{tag}]", fwd, grads, str(dev), tol=NPU_TOL)

    # pooling 也走一遍 NPU
    xs = torch.randn(17, 16)
    bs = torch.tensor([0] * 1 + [1] * 2 + [2] * 5 + [3] * 9)
    w = torch.randn(4, 16)

    xc = xs.clone().detach().requires_grad_(True)
    (global_mean_pool(xc, bs) * w).sum().backward()

    xn = xs.clone().detach().to(dev).requires_grad_(True)
    (ascend_global_mean_pool(xn, bs.to(dev)) * w.to(dev)).sum().backward()

    fwd = (ascend_global_mean_pool(xn, bs.to(dev)).detach().cpu()
           - global_mean_pool(xc, bs).detach()).abs().max().item()
    grads = [(xn.grad.cpu() - xc.grad).abs().max().item()]
    _record("NPU ascend_global_mean_pool", fwd, grads, str(dev), tol=NPU_TOL)


def main():
    torch.manual_seed(2026)
    print("=" * 78)
    print(f"Ascend 纯张量算子 vs PyG 原版算子（容差 {TOL:g}）")
    print("=" * 78)

    print("\n[1] 正常图（多节点多边）")
    x, ei = normal_graph()
    compare_conv(x, ei, "正常图")
    batch = torch.tensor([0] * 1 + [1] * 2 + [2] * 5 + [3] * 9)
    compare_pool(torch.randn(17, 16), batch, "四图批量")

    print("\n[2] reset 极端边界图（1 节点 0 边）")
    xb, eib, extra = reset_graph()
    print(f"  {extra}")
    compare_conv(xb, eib, "1节点0边")
    compare_pool(torch.randn(xb.size(0), 16),
                 torch.zeros(xb.size(0), dtype=torch.long), "单节点单图")

    print("\n[3] 有向非对称图（防止聚合方向写反）")
    src = [0, 0, 1, 2, 3, 4, 5]
    dst = [1, 2, 2, 3, 4, 5, 0]
    ei_dir = torch.tensor([src, dst], dtype=torch.long)
    compare_conv(torch.randn(6, 8), ei_dir, "有向非对称")
    dense_reference_check(src, dst)

    print("\n[4] 边界组 batch（含 1 节点图 + 多节点图）")
    xs = torch.randn(1 + 6, 16)
    bs = torch.tensor([0] + [1] * 6)
    compare_pool(xs, bs, "1+6 节点两图")

    npu_check()

    print("\n" + "=" * 78)
    failed = [r for r in results if not r[3]]
    for name, fwd, grad, ok, _ in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<36s} "
              f"fwd={fwd:.3e} grad={grad:.3e}")
    print("=" * 78)
    if failed:
        print(f"验收失败：{len(failed)} 项超差")
        sys.exit(1)
    print(f"验收通过：{len(results)} 项全部低于各自容差"
          f"（CPU {TOL:g} / NPU {NPU_TOL:g}）")


if __name__ == "__main__":
    main()
