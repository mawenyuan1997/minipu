from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="minipu_backend._C",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="minipu_backend._C",
            sources=[
                "csrc/allocator.cpp",
                "csrc/runtime.cpp",
                "csrc/kernels.cu",
                "csrc/factory.cpp",
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=["my_gpu_backend"],
)
