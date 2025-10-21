"""Centralized device selection utilities.

Provides a single place to select the device (mps/cuda/cpu) and helpers
to move tensors/structures to that device. Use this module in other files
instead of redefining _select_device/global_device logic across the repo.
"""
from typing import Any
try:
    import torch
except Exception as e:
    # Re-raise with clearer message when torch is unavailable at import time.
    raise ImportError("torch is required by utils.device; activate your environment or install torch") from e


def select_device() -> str:
    """Return a short device name: 'mps', 'cuda' or 'cpu'."""
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# module-level globals for convenience
global_device_name: str = select_device()
global_device: torch.device = torch.device(global_device_name)


def to_device(x: Any, device=None) -> Any:
    """Move a tensor/array/list/tuple/dict to the requested device.

    If device is None, uses module-global `global_device`.
    Handles torch.Tensor and numpy arrays (converts to torch tensors).
    """
    if device is None:
        device = global_device

    # accept device string as well
    if isinstance(device, str):
        device = torch.device(device)

    import numpy as _np

    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, _np.ndarray):
        return torch.as_tensor(x, device=device)
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(t, device) for t in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


def print_device():
    print(f"[Device] Using {global_device_name} (MPS available: {getattr(torch.backends, 'mps', None) and getattr(torch.backends, 'mps', None).is_available() if hasattr(torch.backends, 'mps') else False}, CUDA available: {torch.cuda.is_available()})")
