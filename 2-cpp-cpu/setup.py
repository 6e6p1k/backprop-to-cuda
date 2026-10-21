import os, subprocess, sys
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# find scipy_openblas32 (cblas.h + lib)
result = subprocess.run(
    [sys.executable, "-c",
     "import scipy_openblas32, os; print(os.path.dirname(scipy_openblas32.__file__))"],
    capture_output=True, text=True
)
openblas_dir = result.stdout.strip()
openblas_inc = os.path.join(openblas_dir, "include")
openblas_lib = os.path.join(openblas_dir, "lib")

setup(
    ext_modules=[Pybind11Extension(
        "tensor",
        ["bindings.cpp"],
        include_dirs=[openblas_inc],
        library_dirs=[openblas_lib],
        libraries=["libscipy_openblas" if os.name == "nt" else "scipy_openblas"],
        extra_link_args=[] if os.name == "nt" else [f"-Wl,-rpath,{openblas_lib}"],
    )],
    cmdclass={"build_ext": build_ext},
)
