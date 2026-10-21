# 3. Eager CUDA

What changes from step 2: tensors can live on the GPU (`to_cuda()`, `to_cpu()`), and the main forward ops run there.

- **Elementwise:** `add` and `mul` with broadcasting, through one hand-written strided CUDA kernel, plus a fill kernel (`cuda_ops.cu`).
- **Matmul:** cuBLAS strided-batched GEMM (`matmul_fast`, `matmul_cuda`).
- **Reductions:** `sum` through cuTENSOR.

The rest of the code:

- `src/` also holds the C++ autograd with 11 backward node types and the pybind11 bindings (`bindings.cpp`).
- `bobrito/`: the Python wrapper and a small `>>` model API.
- `tests/accuracy_test.py`: 27 checks against PyTorch on the GPU. The recorded timings against PyTorch are in `benchmark_results.txt` and `tests/benchmark_results.txt`.
- Built with CMake against CUDA 13 for `sm_120`. OpenBLAS and cuTENSOR come from the active conda environment.

## What this step couldn't do

- **Only the forward pass moved to the GPU.**
  - The backward nodes still compute on host vectors, as in step 2, so training cannot stay on the GPU.
  - `log`, `exp`, `pow`, `mean`, `max`, `argmax`, `eq`, `reshape`, `squeeze`, `unsqueeze`, `flatten` and `transpose` have no CUDA path yet.
- **Every op is eager.** Each op runs and allocates as soon as it is called, so there is no graph to plan memory or launches from.
- **It is slower than PyTorch.** The recorded matmul timings are 1.3x to 14x slower. The numbers are kept as measured, but the method is loose:
  - `tests/1.py` times CPU inputs, so the host-to-device copy is included, and it does not synchronise.
  - The "mul" cells in `tests/1.ipynb` actually run `a + b`.
- **No ReLU.**

Step 4 starts from these limits: ops build a graph, the backward pass is part of the same graph, and everything runs on the GPU when the result is needed.

The fixes made while collecting this step have not been rebuilt or run on a GPU:

- the leaf `grad_fn` no longer shadows the member;
- the CPU `sum` no longer divides by N;
- the package imports are relative;
- `tsr()` passes its `requires_grad` and `device` arguments through for list and Tensor input;
- `scalar` and `from_torch` are fixed.
