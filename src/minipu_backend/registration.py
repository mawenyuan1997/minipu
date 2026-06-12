import importlib
from typing import List

import torch


BACKEND_NAME = "minipu"
_REGISTERED = False


class _MiniPUDeviceModule:
    """Tiny device module exposed as torch.minipu after registration."""

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def get_device_name(device=None) -> str:
        index = 0 if device is None else torch.device(device).index or 0
        return f"MiniPU:{index}"

    @staticmethod
    def synchronize(device=None) -> None:
        return None

    @staticmethod
    def manual_seed_all(seed: int) -> None:
        torch.manual_seed(seed)

    @staticmethod
    def get_amp_supported_dtype() -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]


def _require_privateuse1_api() -> None:
    missing = [
        name
        for name in ("rename_privateuse1_backend", "_register_device_module")
        if not hasattr(torch, name)
    ]
    if not hasattr(torch.utils, "generate_methods_for_privateuse1_backend"):
        missing.append("torch.utils.generate_methods_for_privateuse1_backend")
    if missing:
        raise RuntimeError(
            "This demo requires a modern PyTorch build with PrivateUse1 helpers. "
            f"Missing: {', '.join(missing)}"
        )


def register() -> None:
    """Load the C++ fallback extension and expose the backend as ``minipu``."""

    global _REGISTERED
    if _REGISTERED:
        return

    _require_privateuse1_api()
    importlib.import_module("minipu_backend._C")

    torch.rename_privateuse1_backend(BACKEND_NAME)
    torch._register_device_module(BACKEND_NAME, _MiniPUDeviceModule)
    torch.utils.generate_methods_for_privateuse1_backend(
        for_tensor=True,
        for_module=True,
        for_storage=True,
    )
    _REGISTERED = True


def is_registered() -> bool:
    return _REGISTERED
