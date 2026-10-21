# 2. C++ on the CPU

The step 1 engine rewritten in C++17:

- `tensor.h`: strided N-d tensors with broadcasting.
- `autograd.h`: a graph of backward nodes with topological ordering.
- `ops.h`: OpenBLAS `sgemm` for the matmul forward and backward passes.
- `main.cpp`: 12 backward tests with hand-computed values.
- `bindings.cpp` and `setup.py`: pybind11 bindings. `tools.py` and `module.py` rebuild the step 1 `>>` API on top, with `LinearBlas`.
- `training.py`: MNIST on the C++ backend. `benchmark.py` compares naive matmul with BLAS matmul.

Build and run:

```
pip install pybind11 scipy_openblas32 numpy torch keras
python setup.py build_ext --inplace
python benchmark.py
python training.py
```

## What this step couldn't do

- **It runs on the CPU only.**
- **The BLAS matmul only handles contiguous inputs**, so it ignores strides and a transposed input gives wrong results.
- **No ReLU**, so `training.py` is a linear model.

Step 3 moves the tensors to the GPU.
