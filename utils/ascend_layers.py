"""昇腾/CPU 可用的纯张量图算子。

目的：彻底去掉对 torch_scatter / torch_sparse / pyg_lib 等 C++/CUDA 扩展的
依赖，同时与 PyG 原版算子数值一致（前向 + 反向）。

- `AscendGCNConv`            : 对齐 torch_geometric.nn.GCNConv
                               （含 add_self_loops、对称度归一化、bias），
                               参数名保持一致（`lin.weight` / `bias`），
                               所以可以直接加载用 PyG 训练出来的权重。
- `ascend_global_mean_pool`  : 对齐 torch_geometric.nn.global_mean_pool。
- `ascend_gcn_norm`          : 复刻 PyG 的 gcn_norm（source_to_target 流）。
"""

import math

import torch
import torch.nn as nn


def _as_long_index(edge_index, device):
    if edge_index is None:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.as_tensor(edge_index, device=device)
    if edge_index.dtype != torch.long:
        edge_index = edge_index.long()
    return edge_index.to(device)


def _add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes):
    """复刻 PyG add_remaining_self_loops：丢掉已有自环，再给每个节点补一条。"""
    if num_nodes == 0:
        empty_e = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        empty_w = edge_weight.new_empty((0,))
        return empty_e, empty_w

    row, col = edge_index[0], edge_index[1]
    mask = row != col
    edge_index = edge_index[:, mask]
    edge_weight = edge_weight[mask]

    loop = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
    loops = torch.stack([loop, loop], dim=0)
    edge_index = torch.cat([edge_index, loops], dim=1)
    loop_weight = torch.full(
        (num_nodes,), float(fill_value), dtype=edge_weight.dtype,
        device=edge_weight.device)
    edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
    return edge_index, edge_weight


def ascend_gcn_norm(edge_index, num_nodes, edge_weight=None,
                    improved=False, add_self_loops=True, dtype=None):
    """对称归一化：D^-1/2 (A + I) D^-1/2，与 PyG gcn_norm 一致。"""
    add_self_loops = bool(add_self_loops)
    fill_value = 2.0 if improved else 1.0

    edge_index = _as_long_index(edge_index, edge_index.device if isinstance(
        edge_index, torch.Tensor) else "cpu")

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype or torch.float32,
                                 device=edge_index.device)
    else:
        edge_weight = edge_weight.to(device=edge_index.device)
        if dtype is not None and edge_weight.dtype != dtype:
            edge_weight = edge_weight.to(dtype)

    if add_self_loops:
        edge_index, edge_weight = _add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index[0], edge_index[1]
    deg = torch.zeros(num_nodes, dtype=edge_weight.dtype, device=edge_index.device)
    deg.index_add_(0, col, edge_weight)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
    norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_index, norm


class AscendGCNConv(nn.Module):
    """GCNConv 的纯张量实现（index_add_ + 对称度归一化）。

    与 PyG 版本的关系：
      out = A_hat_norm @ (X W) + b
      A_hat = A + I（可选），A_hat_norm = D^-1/2 A_hat D^-1/2
    参数命名与 PyG 完全一致（`lin.weight`, `bias`），权重文件可互相加载。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 add_self_loops=True,
                 improved=False,
                 normalize=True,
                 bias=True,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.improved = improved
        self.normalize = normalize

        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # 与 PyG 一致：glorot 初始化权重，bias 置零
        gain = math.sqrt(2.0)
        fan_in = self.in_channels
        fan_out = self.out_channels
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(3.0) * std
        with torch.no_grad():
            self.lin.weight.uniform_(-a, a)
            if self.bias is not None:
                self.bias.zero_()

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.size(0)
        if num_nodes == 0:
            return x.new_zeros((0, self.out_channels))

        edge_index = _as_long_index(edge_index, x.device)

        if self.normalize:
            edge_index, edge_weight = ascend_gcn_norm(
                edge_index, num_nodes, edge_weight,
                improved=self.improved,
                add_self_loops=self.add_self_loops,
                dtype=x.dtype)
        else:
            if edge_weight is None:
                edge_weight = torch.ones(
                    edge_index.size(1), dtype=x.dtype, device=x.device)

        hidden = self.lin(x)

        out = hidden.new_zeros((num_nodes, hidden.size(1)))
        if edge_index.size(1) > 0:
            # source_to_target：把源节点特征按归一化权重累加到目标节点
            msg = hidden[edge_index[0]] * edge_weight.unsqueeze(-1)
            out.index_add_(0, edge_index[1], msg)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return (f"{self.__class__.__name__}({self.in_channels}, "
                f"{self.out_channels})")


def ascend_global_mean_pool(x, batch=None, size=None):
    """global_mean_pool 的纯张量等价实现。"""
    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)
    if not isinstance(batch, torch.Tensor):
        batch = torch.as_tensor(batch, device=x.device)
    batch = batch.to(device=x.device)
    if batch.dtype != torch.long:
        batch = batch.long()

    if size is not None:
        num_graphs = int(size)
    elif batch.numel() == 0:
        num_graphs = 0
    else:
        num_graphs = int(batch.max().item()) + 1

    if num_graphs == 0:
        return x.new_zeros((0, x.size(1)))

    out = x.new_zeros((num_graphs, x.size(1)))
    out.index_add_(0, batch, x)

    counts = torch.zeros(num_graphs, dtype=x.dtype, device=x.device)
    counts.index_add_(0, batch, torch.ones(
        batch.size(0), dtype=x.dtype, device=x.device))
    return out / counts.clamp(min=1.0).unsqueeze(-1)


def ascend_global_add_pool(x, batch=None, size=None):
    """可选的 add_pool 等价实现（SAC 的 log_prob 汇总会用）。"""
    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)
    batch = torch.as_tensor(batch, device=x.device).long()
    num_graphs = int(size) if size is not None else (
        int(batch.max().item()) + 1 if batch.numel() else 0)
    if num_graphs == 0:
        return x.new_zeros((0, x.size(1)))
    out = x.new_zeros((num_graphs, x.size(1)))
    out.index_add_(0, batch, x)
    return out
