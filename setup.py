from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "fast_algo",
        ["cpp_core/fast_algo.cpp"],
    ),
]

setup(
    name="fast_algo",
    version="1.0.0",
    description="High-performance analytics algorithms in C++",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
    options={
        'build_ext': {
            'plat_name': 'win-amd64',
        }
    },
)
