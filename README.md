# MiniPU PrivateUse1 Backend

MiniPU demonstrates PyTorch `PrivateUse1` registration and a runnable custom-operator codegen script. The custom operators accept CPU tensors, generate CUDA code, and use ATen CPU operators to compute results. Separate `PrivateUse1` ATen handlers are registered, but we can't have real MiniPU tensors because device allocation and copy support are not implemented.

## Install and run
```
python -m pip install -e .
python examples/smoke.py
```

## Registering The Backend

`examples/smoke.py` demostrates how it works. The code

```python
import minipu_backend

minipu_backend.register()
```
 does four things in `register()` of `src/minipu_backend/registration.py`:

1. Verifies that the installed PyTorch has the required `PrivateUse1` helpers.
2. Imports `minipu_backend._C`, which loads the C++ dispatcher registrations.
3. Renames `PrivateUse1` to `minipu` with `torch.rename_privateuse1_backend`.
4. Registers a small `torch.minipu` module and asks PyTorch to generate helper
   methods such as `Tensor.minipu()` and `Tensor.is_minipu`.

After registration, Python code can refer to the backend by the name `minipu`
instead of the internal dispatch key name `PrivateUse1`.

## PyTorch Runtime Flow

```python
x = torch.tensor([-2.0, 1.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
```

PyTorch creates two normal CPU tensor instead of minipu tensor because in this project, we didn't implement functionalities like allocating memory on device and copying the tensor to device. Creating them with `device="minipu"` would fail

```python
added = torch.ops.minipu.add(x, y)
```

This line calls the custom PyTorch operator named `minipu::add`. Since x and y are CPU tensors, PyTorch selects its registered CPU implementation `minipu_custom_add`. That function generates `add.cu` and computes the result using PyTorch’s CPU operator `at::add`.


## CUDA Code Generation

When an operator is called, PyTorch’s dispatcher identifies operator as `minipu::add` and tensor key as `CPU`. It finds

```cpp
TORCH_LIBRARY_IMPL(minipu, CPU, m) {
  m.impl("add", TORCH_FN(minipu_custom_add));
}
```

from `src/minipu_backend/csrc/backend.cpp` and then it calls `minipu_custom_add(x, y)` which generates the CUDA code and computes the result on CPU
