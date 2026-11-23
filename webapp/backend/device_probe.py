"""设备探测：复用仓库 utils.device，但不直接 import torch。

策略：在 REPO_ROOT 下 subprocess 调用 `python -c "from utils.device import device_name; print(device_name('auto'))"`。
- 成功 → 解析 npu:0 / cuda:0 / cpu。
- 失败（无 torch / 无 utils）→ 回退 cpu。

这样 webapp 自身不依赖 torch，本地与 ECS 都能零报错启动。
"""
from __future__ import annotations

import subprocess
import sys
import os
import logging

from .config import REPO_ROOT, DEVICE_FORCE
from .schemas import DeviceInfo

_logger = logging.getLogger("webapp.device")
_cached: DeviceInfo | None = None


def probe_device() -> DeviceInfo:
    global _cached
    if _cached is not None:
        return _cached

    if DEVICE_FORCE:
        dev = DEVICE_FORCE.strip().lower()
        npu_avail = "npu" in dev
        _cached = DeviceInfo(
            type=dev if dev in ("npu:0", "cuda:0", "cpu") else "cpu",
            npu_available=npu_avail,
            label="Ascend NPU" if npu_avail else ("CUDA" if "cuda" in dev else "CPU"),
        )
        _logger.info("device forced to %s", _cached.type)
        return _cached

    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "from utils.device import device_name; print(device_name('auto'))"],
            cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=10,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        )
        out = result.stdout.strip().lower()
        if result.returncode == 0 and out:
            npu_avail = "npu" in out
            label = "Ascend NPU" if npu_avail else ("CUDA" if "cuda" in out else "CPU")
            _cached = DeviceInfo(
                type=out if out in ("npu:0", "cuda:0", "cpu") else "cpu",
                npu_available=npu_avail, label=label,
            )
            _logger.info("device probed: %s", _cached.type)
            return _cached
        _logger.warning("device probe subprocess failed: rc=%s stderr=%s",
                        result.returncode, result.stderr[:200])
    except Exception as exc:
        _logger.warning("device probe exception: %s", exc)

    _cached = DeviceInfo(type="cpu", npu_available=False, label="CPU")
    _logger.info("device fallback: cpu")
    return _cached
