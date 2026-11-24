#!/usr/bin/env bash
# =============================================================================
# EV-GNN 华为昇腾（ModelArts Ascend 预置镜像 / Linux aarch64）一键部署
#
# 设计原则：
#   1. 绝不动镜像自带的 torch / torch_npu —— 全程用 constraints 文件把它们锁死，
#      既不 `pip install -r requirements.txt`，也不装任何会反向依赖 torch 重装的包。
#   2. ev2gym / torch_geometric 用 --no-deps 安装，避免把 gurobipy、stable-baselines3、
#      torch_scatter / torch_sparse / pyg_lib 这些在 aarch64 上极易编译失败的包带进来。
#   3. 只补纯 Python / 数值依赖。
#
# 用法：
#   bash setup_ascend.sh                 # 在 EV-GNN 仓库根目录执行
#   REPO_DIR=/home/ma-user/work/EV-GNN bash setup_ascend.sh
#   SKIP_DEPS=1 bash setup_ascend.sh     # 只做环境校验，不装包
#   PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash setup_ascend.sh
# =============================================================================
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
PY="${PY:-python3}"
SKIP_DEPS="${SKIP_DEPS:-0}"
CONSTRAINTS="${CONSTRAINTS:-/tmp/evgnn_ascend_constraints.txt}"

log()  { printf '\n\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 0. 预检
log "0/7 预检：架构、Python、仓库"
ARCH="$(uname -m)"
echo "  架构      : ${ARCH}"
[ "${ARCH}" = "aarch64" ] || warn "当前不是 aarch64（本脚本按 ModelArts 昇腾镜像写的），继续但可能不适用。"
echo "  Python    : $(${PY} -V 2>&1)"
echo "  仓库目录  : ${REPO_DIR}"
[ -f "${REPO_DIR}/train_RL_GNN.py" ] || die "在 ${REPO_DIR} 找不到 train_RL_GNN.py，请用 REPO_DIR 指向 EV-GNN 仓库根目录。"

# ---------------------------------------------------------------- 1. 昇腾环境变量
log "1/7 加载 Ascend 环境变量"
for CANDIDATE in \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/nnal/atb/set_env.sh \
    /usr/local/Ascend/driver/bin/setenv.bash; do
  if [ -f "${CANDIDATE}" ]; then
    # shellcheck disable=SC1090
    set +u; source "${CANDIDATE}"; set -u
    echo "  sourced: ${CANDIDATE}"
  fi
done
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
  warn "ASCEND_HOME_PATH 仍为空，若 npu-smi / torch.npu 不可用，请确认镜像里 set_env.sh 的实际路径。"
else
  echo "  ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
fi

# ---------------------------------------------------------------- 2. 设备自检
log "2/7 NPU 自检"
if command -v npu-smi >/dev/null 2>&1; then
  # 先完整取回再 head，避免 head 提前关闭管道让 npu-smi 收到 SIGPIPE（pipefail 下会误报失败）
  NPU_INFO="$(timeout 20 npu-smi info 2>/dev/null || true)"
  if [ -n "${NPU_INFO}" ]; then
    printf '%s\n' "${NPU_INFO}" | head -n 20
  else
    warn "npu-smi info 无输出（容器内可能读不到设备），继续用 torch.npu 判定。"
  fi
else
  warn "未找到 npu-smi（容器里不一定装），继续用 torch.npu 判定。"
fi

"${PY}" - <<'PYEOF'
import sys
try:
    import torch
except Exception as exc:
    print(f"[error] 无法 import torch: {exc}"); sys.exit(1)

print(f"  torch       : {torch.__version__}")
try:
    import torch_npu
    print(f"  torch_npu   : {torch_npu.__version__}")
except Exception as exc:
    print(f"[error] 无法 import torch_npu: {exc}"); sys.exit(1)

ok = bool(hasattr(torch, "npu") and torch.npu.is_available())
print(f"  torch.npu.is_available() = {ok}")
if not ok:
    print("[error] NPU 不可用：检查驱动 / set_env.sh / 容器是否申请到 NPU 卡。")
    sys.exit(2)
print(f"  npu 设备数  : {torch.npu.device_count()}")
PYEOF

# ---------------------------------------------------------------- 3. 锁死镜像自带的 torch
log "3/7 生成 constraints（锁死 torch / torch_npu，防止被依赖重装）"
TORCH_VER="$(${PY} -c 'import torch;print(torch.__version__)')"
NPU_VER="$(${PY} -c 'import torch_npu;print(torch_npu.__version__)')"
{
  echo "torch==${TORCH_VER}"
  echo "torch_npu==${NPU_VER}"
  # 镜像里通常还有这些同源包，一并锁住
  for PKG in torchvision torchaudio; do
    V="$(${PY} -c "import importlib.metadata as m; print(m.version('${PKG}'))" 2>/dev/null | tail -n 1 || true)"
    if [ -n "${V}" ]; then
      echo "${PKG}==${V}"
    fi
  done
} > "${CONSTRAINTS}"
cat "${CONSTRAINTS}" | sed 's/^/  /'

# 按镜像 Python 版本选 numba（0.61+ 只支持 3.10+）
PY_MINOR="$(${PY} -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "${PY_MINOR}" in
  3.9)      NUMBA_PIN="numba==0.60.0" ;;
  3.10|3.11|3.12) NUMBA_PIN="numba==0.61.2" ;;
  *)        NUMBA_PIN="numba>=0.58,<0.62" ;;
esac
echo "  Python ${PY_MINOR} → ${NUMBA_PIN}"

# ---------------------------------------------------------------- 4. 纯 Python / 数值依赖
if [ "${SKIP_DEPS}" = "1" ]; then
  log "4/7 跳过依赖安装（SKIP_DEPS=1）"
else
  log "4/7 安装数值依赖（不触碰 torch 系）"
  "${PY}" -m pip install --no-cache-dir -c "${CONSTRAINTS}" \
      "numpy==1.24.4" \
      "pandas==2.0.3" \
      "${NUMBA_PIN}" \
      "networkx==3.0" \
      "gymnasium==1.0.0" \
      "scipy<1.16" \
      "matplotlib<3.11" \
      "pyyaml" \
      "pandapower==3.1.2" \
      "multicopula==0.0.2" \
      "tqdm" \
      "wandb" \
      "icecream" \
      "setuptools<81"          # ev2gym 1.0.0 里还在用 pkg_resources
fi

# ---------------------------------------------------------------- 5. 装 EV2Gym / PyG（--no-deps）
if [ "${SKIP_DEPS}" != "1" ]; then
  log "5/7 安装 ev2gym==1.0.0 与 torch_geometric==2.5.3（--no-deps）"
  # 注意：必须 1.0.0 —— EV-GNN 的配置文件是 1.0.0 时代的，
  # EV2Gym 主线 2.0.0 会缺失 simulate_grid / min_emergency_battery_capacity 等键。
  "${PY}" -m pip install --no-cache-dir --no-deps "ev2gym==1.0.0"
  "${PY}" -m pip install --no-cache-dir --no-deps "torch_geometric==2.5.3"
  # 明确不装（列出仅作说明，脚本不会执行）：
  #   gurobipy / stable-baselines3 / sb3-contrib
  #   torch_scatter / torch_sparse / pyg_lib
else
  log "5/7 跳过 ev2gym / PyG 安装（SKIP_DEPS=1）"
fi

# ---------------------------------------------------------------- 6. 回验：torch 未被改动
log "6/7 回验：torch / torch_npu 版本未被改动"
AFTER_TORCH="$(${PY} -c 'import torch;print(torch.__version__)')"
AFTER_NPU="$(${PY} -c 'import torch_npu;print(torch_npu.__version__)')"
echo "  torch: ${TORCH_VER} → ${AFTER_TORCH}"
echo "  torch_npu: ${NPU_VER} → ${AFTER_NPU}"
[ "${AFTER_TORCH}" = "${TORCH_VER}" ] || die "torch 版本被改动了（${TORCH_VER} → ${AFTER_TORCH}），请回滚镜像或检查依赖。"
[ "${AFTER_NPU}" = "${NPU_VER}" ]   || die "torch_npu 版本被改动了（${NPU_VER} → ${AFTER_NPU}），请回滚镜像。"

"${PY}" - <<'PYEOF'
import torch, torch_npu
a = torch.randn(64, 64).npu()
b = a @ a
print(f"  昇腾矩阵乘自检: ok, sum={b.sum().item():.4f}, device={b.device}")

# 确认扩展类算子没有被引入（没有 torch_scatter / torch_sparse / pyg_lib）
for mod in ("torch_scatter", "torch_sparse", "pyg_lib"):
    try:
        __import__(mod)
        print(f"  [warn] 检测到 {mod}，本项目改造后并不需要它")
    except Exception:
        print(f"  {mod}: 未安装（符合预期）")
PYEOF

# ---------------------------------------------------------------- 7. 项目自检
log "7/7 项目侧自检"
cd "${REPO_DIR}"
"${PY}" - <<'PYEOF'
import sys

# SKIP_DEPS=1 时可能还没装 ev2gym / PyG，这里只报状态不中断
for name in ("torch", "torch_geometric", "ev2gym"):
    try:
        mod = __import__(name)
        extra = f" {getattr(mod, '__version__', '')}".rstrip()
        print(f"  {name:<15}: ok{(' ' + extra) if extra else ''}")
    except Exception as exc:
        print(f"  [warn] {name} 暂不可用（SKIP_DEPS=1 时属正常）: {exc}")

try:
    from utils.device import get_device
    from utils.ascend_layers import AscendGCNConv, ascend_global_mean_pool
    print(f"  get_device('auto') -> {get_device('auto')}")
    print("  AscendGCNConv / ascend_global_mean_pool 导入成功")
except Exception as exc:
    print(f"  [error] 适配层导入失败: {exc}")
    sys.exit(1)
PYEOF

EQUIV=""
for CAND in "${REPO_DIR}/test_ascend_equiv.py" "${REPO_DIR}/../test_ascend_equiv.py"; do
  if [ -f "${CAND}" ]; then
    EQUIV="${CAND}"
    break
  fi
done
if [ -n "${EQUIV}" ]; then
  echo "  运行算子等价性验收：${EQUIV}"
  ( cd "$(dirname "${EQUIV}")" && "${PY}" "${EQUIV}" ) \
      || warn "等价性测试未通过，请勿直接开始训练。"
fi

cat <<EOF

部署完成。下一步：

  cd ${REPO_DIR}
  # 1) 先确认设备解析
  ${PY} -c "from utils.device import get_device; print(get_device('auto'))"

  # 2) 500 步冒烟（DEVELOPMENT=True 时）
  ${PY} train_RL_GNN.py --policy TD3_ActionGNN --config PublicPST.yaml \\
      --device npu:0 --max_timesteps 500

  # 3) 正式训练：把 train_RL_GNN.py 里 DEVELOPMENT 改回 False，再去掉 --max_timesteps
EOF
