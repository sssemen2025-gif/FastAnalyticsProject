from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "sorting",               # имя модуля (должно совпадать с PYBIND11_MODULE)
        ["cpp_core/OOP.cpp"],         # твой C++ файл
        cxx_std=17                # стандарт C++
    ),
]

setup(
    name="OOP",
    version="1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)