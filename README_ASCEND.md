# README_ASCEND — EV-GNN 在华为昇腾（ModelArts aarch64 镜像）上跑起来

本目录已完成的改造已在本机 CPU 上验证通过：算子等价（误差 < 1e-5）+ 500 步训练/评估闭环无报错。

## 0. 文件在哪

三个昇腾相关文件都在仓库根目录（分支 `feat/ascend-npu`）：

```
EV-GNN/
├── setup_ascend.sh         # 一键部署脚本
├── test_ascend_equiv.py    # 算子等价性验收（CPU 7 项 + NPU 条件项）
├── README_ASCEND.md        # 本文件
└── utils/device.py         # 设备选择
    utils/ascend_layers.py  # 纯张量 GCN / pooling
```

改造范围：`utils/device.py`、`utils/ascend_layers.py` 两个新文件，加上 `TD3/`、`SAC/`、`evaluator.py`、`train_RL_GNN.py`、`utils/replay_buffer.py` 的改动。

## 1. 一键部署

```bash
# 方案 A：直接 clone 本分支（推荐）
cd ~/work
git clone -b feat/ascend-npu <你的 EV-GNN 仓库地址> EV-GNN
cd EV-GNN && bash setup_ascend.sh          # 脚本就在仓库根目录

# 方案 B：脚本在别处 / 只做环境自检 / 走内网源
REPO_DIR=~/work/EV-GNN bash setup_ascend.sh
SKIP_DEPS=1 bash setup_ascend.sh           # 只校验环境与适配层导入，不装任何包
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash setup_ascend.sh
```

脚本会依次做：加载 `set_env.sh` → `npu-smi info` + `torch.npu.is_available()` → 生成 constraints 锁死 `torch`/`torch_npu` → 装数值依赖 → `--no-deps` 装 `ev2gym==1.0.0` 与 `torch_geometric==2.5.3` → 回验 torch 版本没变 → 昇腾矩阵乘自检 → 项目导入自检 → 若同目录有 `test_ascend_equiv.py` 则自动跑算子验收。

## 2. 跑训练

```bash
cd ~/work/EV-GNN

# 设备解析自检（应打印 npu:0）
python -c "from utils.device import get_device; print(get_device('auto'))"

# 500 步冒烟（train_RL_GNN.py 里 DEVELOPMENT=True 时，eval 只跑 2 episode）
python train_RL_GNN.py --policy TD3_ActionGNN --config PublicPST.yaml \
    --device npu:0 --max_timesteps 500

# 正式训练：把 DEVELOPMENT 改回 False，去掉 --max_timesteps
python train_RL_GNN.py --policy TD3_ActionGNN --config PublicPST.yaml --device npu:0
```

`--device` 支持 `auto | cpu | cuda:0 | npu:0`；`auto` 依次探测 NPU → CUDA → CPU，且探测到 NPU 时会尝试 `torch_npu.contrib.transfer_to_npu`。

## 3. 三条不能碰的红线

1. **不要 `pip install -r requirements.txt`。** 会把 `torch_scatter`/`torch_sparse`/`pyg_lib`/`gurobipy` 一起拉进来，aarch64 上基本编译不过，还有覆盖镜像自带 torch 的风险。
2. **ev2gym 必须 1.0.0。** EV-GNN 的 yaml 是 1.0.0 时代的；主线 2.0.0 会 `KeyError: simulate_grid`，且 `ev.desired_capacity` 从 kWh 改成 0~1 比例，语义已变。
3. **不要自己 `pip install torch/torch_npu`。** 镜像里的版本跟驱动绑定，脚本用 constraints 文件锁死了这两个包，装完会回验版本。

## 4. 已剔除的依赖与影响

| 剔除 | 影响 |
| --- | --- |
| `gurobipy` | `evaluator.py` 的 MPC / Optimal 基线不可用（已改成占位，调用时报明确提示），RL 策略评估不受影响 |
| `stable-baselines3` / `sb3-contrib` | 只有 `train_baselines.py` 与 evaluator 的 PPO/SAC 基线需要，同样改成占位 |
| `torch_scatter` / `torch_sparse` / `pyg_lib` | 已用 `AscendGCNConv` + `ascend_global_mean_pool` 顶替，不再需要 |

## 5. 代码改造清单

- `utils/device.py`：`get_device(prefer)` / `as_device()`，NPU→CUDA→CPU 自动回退。
- `utils/ascend_layers.py`：`AscendGCNConv`（`index_add_` + 对称度归一化）、`ascend_global_mean_pool`、`ascend_gcn_norm`。参数名与 PyG 完全一致（`lin.weight` / `bias`），**旧权重可直接加载**。
- `TD3/`、`SAC/`、`evaluator.py`、`utils/replay_buffer.py`：`cuda` 硬编码 → `get_device()`；`GCNConv`/`global_mean_pool` → Ascend 版；`torch.load` 全部补 `map_location`。
- `train_RL_GNN.py`：`--device` 支持 `auto|cpu|cuda:0|npu:0`；`os.system('cp ...')` → `shutil.copy2`（源文件缺失只告警）；三个 ReplayBuffer 构造时显式传 `device=device`，避免"模型在 CPU、采样张量在 NPU"的跨设备报错。
- `TD3_ActionGNN.py` / `TD3_GNN.py` / `SAC/model.py`：四类节点索引赋值加空张量防护（reset 第 0 步只有 1 个 env 节点、0 条边、`ev_indexes` 为空）。

## 6. 本机验收结果（CPU，Python 3.10.22，torch 2.2.2）

```
AscendGCNConv[正常图]        fwd=0.000e+00  grad=0.000e+00   PASS
ascend_global_mean_pool[四图] fwd=0.000e+00  grad=0.000e+00   PASS
AscendGCNConv[1节点0边]      fwd=0.000e+00  grad=0.000e+00   PASS   ← 真实 reset 状态
ascend_global_mean_pool[单节点] fwd=0.000e+00 grad=0.000e+00  PASS
AscendGCNConv[有向非对称]     fwd=0.000e+00  grad=0.000e+00   PASS
稠密公式复核                 fwd=2.980e-07                PASS   ← 不经过 PyG 的独立校验
验收通过：7 项全部低于各自容差（CPU 1e-5 / NPU 1e-4）
```

脚本第 5 段是**条件项**：`get_device("auto").type == "npu"` 时会把算子与「17 节点正常图」「1 节点 0 边边界图」搬到 `npu:0` 跑前向+反向，与 CPU 结果比对（容差 1e-4）；没有 NPU 时只打印一行「跳过」。所以上华为云后一条 `bash setup_ascend.sh` 或一条 `python test_ascend_equiv.py` 就能验完 NPU 算子。

`TD3_ActionGNN` 500 步（`--device cpu`、`DEVELOPMENT=True`）：4 个 episode + 2 次 eval + Final Evaluation 全部正常，权重落在 `saved_models/<实验名>/`。

## 7. 常见问题

- **`torch.npu.is_available()` 是 False**：`set_env.sh` 没生效或容器没申请 NPU 卡；`source /usr/local/Ascend/ascend-toolkit/set_env.sh` 后重开 shell 试。
- **numba 装不上**：镜像 Python 3.9 要 `numba==0.60.0`（脚本已按版本自动选）。
- **`npu-smi` 不存在**：容器里常见，不影响，`torch.npu` 判定为准。
- **gymnasium 打印 "obs 不是 numpy array"**：观测本来就是 PyG `Data`，无害。
- **`pkg_resources is deprecated`**：ev2gym 1.0.0 老代码，已锁 `setuptools<81`，无害。
