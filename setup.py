from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


ROOT = Path(__file__).parent


setup(
    ext_modules=[
        CppExtension(
            name="minipu_backend._C",
            sources=[str(ROOT / "src" / "minipu_backend" / "csrc" / "backend.cpp")],
            extra_compile_args={"cxx": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
