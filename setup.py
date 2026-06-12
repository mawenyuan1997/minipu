from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    ext_modules=[
        CppExtension(
            name="minipu_backend._C",
            sources=["src/minipu_backend/csrc/backend.cpp"],
            extra_compile_args={"cxx": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
