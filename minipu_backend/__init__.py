import torch

from . import _C


BACKEND_NAME = "minipu"


def _rename_privateuse1_backend(name: str) -> None:
    rename = getattr(
        torch,
        "rename_privateuse1_backend",
        None,
    )

    if rename is None:
        rename = getattr(
            torch.utils,
            "rename_privateuse1_backend",
            None,
        )

    if rename is None:
        raise RuntimeError(
            "PrivateUse1 backend rename API is unavailable"
        )

    rename(name)

class _MiniPUDeviceModule:
    """Python device API exposed as torch.minipu."""

    @staticmethod
    def is_available() -> bool:
        return torch.cuda.is_available()

    @staticmethod
    def device_count() -> int:
        return torch.cuda.device_count()

    @staticmethod
    def current_device() -> int:
        return torch.cuda.current_device()

    @staticmethod
    def set_device(device) -> None:
        index = torch.device(device).index

        if index is None:
            index = 0

        torch.cuda.set_device(index)

    @staticmethod
    def synchronize(device=None) -> None:
        if device is None:
            torch.cuda.synchronize()
            return

        index = torch.device(device).index

        if index is None:
            index = 0

        torch.cuda.synchronize(index)

    @staticmethod
    def get_device_name(device=None) -> str:
        if device is None:
            index = torch.cuda.current_device()
        else:
            index = torch.device(device).index
            if index is None:
                index = 0

        return torch.cuda.get_device_name(index)

    @staticmethod
    def manual_seed_all(seed: int) -> None:
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def get_amp_supported_dtype():
        return [
            torch.float16,
            torch.bfloat16,
        ]

    @staticmethod
    def _is_in_bad_fork() -> bool:
        return False


_rename_privateuse1_backend(BACKEND_NAME)

torch._register_device_module(
    BACKEND_NAME,
    _MiniPUDeviceModule,
)

torch.utils.generate_methods_for_privateuse1_backend(
    for_tensor=True,
    for_module=True,
    for_storage=True,
)