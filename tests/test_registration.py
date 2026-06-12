import importlib.util
from pathlib import Path

import pytest
import torch


def _missing_privateuse1_api():
    missing = []
    if not (
        hasattr(torch, "rename_privateuse1_backend")
        or hasattr(torch.utils, "rename_privateuse1_backend")
    ):
        missing.append("rename_privateuse1_backend")
    if not hasattr(torch, "_register_device_module"):
        missing.append("_register_device_module")
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


@pytest.mark.skipif(
    not _has_built_extension(),
    reason="C++ extension is not built; run `python -m pip install -e .`",
)
def test_emit_demo_codegen(tmp_path, monkeypatch):
    import minipu_backend

    monkeypatch.setenv("MINIPU_CODEGEN_DIR", str(tmp_path))

    out_dir = Path(minipu_backend.emit_demo_codegen())

    assert out_dir == tmp_path
    assert sorted(path.name for path in out_dir.glob("*.cu")) == [
        "add.cu",
        "mul.cu",
        "relu.cu",
    ]


@pytest.mark.skipif(
    not _has_built_extension(),
    reason="C++ extension is not built; run `python -m pip install -e .`",
)
def test_custom_ops_run_and_generate_cuda(tmp_path, monkeypatch):
    import minipu_backend

    monkeypatch.setenv("MINIPU_CODEGEN_DIR", str(tmp_path))
    minipu_backend.register()

    x = torch.tensor([-2.0, 1.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])

    assert torch.ops.minipu.add(x, y).tolist() == [2.0, 6.0, 9.0]
    assert torch.ops.minipu.mul(x, y).tolist() == [-8.0, 5.0, 18.0]
    assert torch.ops.minipu.relu(x).tolist() == [0.0, 1.0, 3.0]
    assert sorted(path.name for path in tmp_path.glob("*.cu")) == [
        "add.cu",
        "mul.cu",
        "relu.cu",
    ]


@pytest.mark.skipif(
    not _has_built_extension(),
    reason="C++ extension is not built; run `python -m pip install -e .`",
)
def test_custom_ops_reject_inputs_outside_codegen_contract():
    import minipu_backend

    minipu_backend.register()

    with pytest.raises(RuntimeError, match="torch.float32"):
        torch.ops.minipu.add(torch.tensor([1]), torch.tensor([2]))
