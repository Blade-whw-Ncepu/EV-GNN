"""设备选择：Ascend NPU → CUDA → CPU，自动回退。

- `get_device(prefer)`:  prefer 支持 "auto" / "cpu" / "cuda" / "cuda:0" /
  "npu" / "npu:0"。auto 时优先 torch_npu，其次 cuda，最后 cpu。
- 探测到昇腾时，会尽力导入 `torch_npu.contrib.transfer_to_npu`，让残留的
  `torch.cuda.*` 调用自动改道到 NPU（失败的场景不影响主流程）。
"""

import os
import warnings

import torch

_NPU_AVAILABLE = None
_TRANSFER_ENABLED = None


def _probe_npu():
    """只在第一次调用时真的去 import torch_npu / 询问 torch.npu。"""
    global _NPU_AVAILABLE
    if _NPU_AVAILABLE is not None:
        return _NPU_AVAILABLE
    try:
        import torch_npu  # noqa: F401
        _NPU_AVAILABLE = bool(hasattr(torch, "npu") and torch.npu.is_available())
    except Exception:
        _NPU_AVAILABLE = False
    return _NPU_AVAILABLE


def enable_transfer_to_npu():
    """把 torch.cuda.* 重定向到 NPU。仅在检测到 NPU 时尝试。"""
    global _TRANSFER_ENABLED
    if _TRANSFER_ENABLED is not None:
        return _TRANSFER_ENABLED
    if os.environ.get("EVGNN_NO_TRANSFER_TO_NPU", "0") == "1":
        _TRANSFER_ENABLED = False
        return False
    try:
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
        _TRANSFER_ENABLED = True
    except Exception:
        _TRANSFER_ENABLED = False
    return _TRANSFER_ENABLED


def _cuda_available():
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def get_device(prefer="auto"):
    """返回 torch.device。无法满足 prefer 时逐级回退并给出提示。"""
    prefer = "auto" if prefer is None else str(prefer).strip().lower()
    npu = _probe_npu()

    if prefer in ("auto", ""):
        if npu:
            enable_transfer_to_npu()
            return torch.device("npu:0")
        if _cuda_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    if prefer.startswith("npu"):
        if not npu:
            warnings.warn("请求了 NPU 但 torch.npu.is_available() 为 False，回退。")
            return get_device("cuda" if _cuda_available() else "cpu")
        enable_transfer_to_npu()
        return torch.device(prefer if ":" in prefer else "npu:0")

    if prefer.startswith("cuda"):
        if not _cuda_available():
            warnings.warn("请求了 CUDA 但不可用，回退到 CPU。")
            return torch.device("cpu")
        return torch.device(prefer if ":" in prefer else "cuda:0")

    if prefer == "cpu":
        return torch.device("cpu")

    warnings.warn(f"未知设备 '{prefer}'，按 auto 处理。")
    return get_device("auto")


def as_device(device):
    """torch.device / str / None 统一成 torch.device。"""
    if isinstance(device, torch.device):
        return device
    return get_device("auto" if device is None else device)


def device_name(prefer="auto"):
    """便于日志打印：返回 'cpu' / 'cuda:0' / 'npu:0'。"""
    return str(get_device(prefer))


def to_device_module(module, device=None):
    """把 nn.Module 搬到解析后的设备上。"""
    return module.to(as_device(device))
