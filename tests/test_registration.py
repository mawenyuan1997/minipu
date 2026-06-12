import importlib.util

import pytest
import torch


def _missing_privateuse1_api():
    missing = [
        name
        for name in ("rename_privateuse1_backend", "_register_device_module")
        if not hasattr(torch, name)
    ]
    if not hasattr(torch.utils, "generate_methods_for_privateuse1_backend"):
        missing.append("torch.utils.generate_methods_for_privateuse1_backend")
    return missing


def _has_built_extension():
    try:
        return importlib.util.find_spec("minipu_backend._C") is not None
    except ModuleNotFoundError:
        return False


@pytest.mark.skipif(
    bool(_missing_privateuse1_api()),
    reason="installed PyTorch does not expose modern PrivateUse1 helpers",
)
@pytest.mark.skipif(
    not _has_built_extension(),
    reason="C++ extension is not built; run `python -m pip install -e .`",
)
def test_register_exposes_backend_module():
    import minipu_backend

    minipu_backend.register()

    assert minipu_backend.is_registered()
    assert torch.minipu.is_available()
    assert torch.minipu.device_count() == 1
    assert hasattr(torch.Tensor, "minipu")
