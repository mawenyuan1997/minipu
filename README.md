# MiniPU PrivateUse1 Backend

`minipu_backend` is a minimal PyTorch custom backend skeleton that uses the
reserved `PrivateUse1` dispatch key. It does not implement any custom kernels.
Instead, it registers a boxed backend fallback that sends unsupported operators
through PyTorch's CPU fallback path.

This is meant to be a small starting point for experimenting with out-of-tree
backends, not a real accelerator runtime.

## What It Contains

- A C++ extension that registers:
  - `PrivateUse1` boxed fallback -> `at::native::cpu_fallback`
  - `AutocastPrivateUse1` fallthrough
- A Python registration helper that:
  - imports the C++ extension
  - renames `PrivateUse1` to `minipu`
  - registers a tiny `torch.minipu` device module
  - generates `Tensor.minipu()`, `Tensor.is_minipu`, and related helpers
- A smoke example and pytest coverage.

## Requirements

- Python 3.9+
- PyTorch 2.1+ recommended
- A working C++ compiler compatible with your PyTorch install

The default Python in this workspace has PyTorch 1.2.0, which predates the
modern `PrivateUse1` APIs. The tests therefore skip on that interpreter.

## Install

```bash
python -m pip install -e .
```

## Run The Example

```bash
python examples/smoke.py
```

Expected output on a modern PyTorch install:

```text
backend: minipu
extension registered: True
torch.minipu.is_available(): True
```

If your PyTorch build supports creating `PrivateUse1` tensors with CPU fallback
alone, the example will also run a tiny tensor operation. Many real backends
still need at least allocator, storage, copy, generator, and device guard pieces
before tensors can live on the renamed device.

## Useful References

- PyTorch PrivateUse1 tutorial:
  https://docs.pytorch.org/tutorials/advanced/privateuseone.html
- PyTorch C++ extension docs:
  https://pytorch.org/tutorials/advanced/cpp_extension.html
