# MiniPU PrivateUse1 Backend

MiniPU demonstrates PyTorch `PrivateUse1` registration using a GPU backend. 

## Components

- `allocator.cpp` implements `c10::Allocator` using CUDA memory APIs.
- `runtime.cpp` registers the `PrivateUse1` DeviceGuard implementation.
- `factory.cpp` registers `empty.memory_format` and `empty_strided`.
- `kernels.cu` implements and registers CUDA kernels (currently only has add).
- `setup.py` compiles C source files into `_C.so` and `TORCH_LIBRARY_IMPL` registers their functions with the dispatcher when the extension is loaded.
- `run_test.py` creates two empty tensors and perform add on GPU
- `minipu_backend` renames 'privateuse1' to 'minipu', defines and registers a Python device module for 'minipu'
## Tested Environment
```text
GPU: NVIDIA GeForce RTX 5060 Ti
Compute capability: 12.0 (sm_120)
CUDA Toolkit: 12.8
PyTorch: 2.11.0+cu128
Python: 3.12
```
Commands to run
```
python -m pip install -e . --no-build-isolation -v
python run_test.py
```
Results:
```text
Custom Accelerator Available: True
Tensor X device: minipu:0
Done
```
