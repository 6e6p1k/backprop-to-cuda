# backprop-to-cuda

From learning how backpropagation works to [Bobrito](https://github.com/6e6p1k/bobrito), a deep learning framework in C++ and CUDA, in four steps.

| Step | Folder | Written | What it is |
|---|---|---|---|
| 1 | [`1-numpy-autograd`](1-numpy-autograd) | 21 to 23 Mar 2026 | Theory notes, a micrograd follow-along, autograd over NumPy checked against PyTorch, and model-definition experiments trained on MNIST |
| 2 | [`2-cpp-cpu`](2-cpp-cpu) | 24 Mar 2026 | The same engine in C++17 on the CPU, with OpenBLAS and pybind11 |
| 3 | [`3-eager-cuda`](3-eager-cuda) | 24 to 29 Mar 2026 | Eager tensors on the GPU, using CUDA kernels, cuBLAS and cuTENSOR |
| 4 | [`4-bobrito`](4-bobrito) | from Apr 2026 | A lazy static-graph framework, in its own repository |

Steps 1 to 3 are collected from two earlier repositories and cleaned up. Scratch files, build output and commented-out code were removed. Bugs found while collecting them were fixed. Each step's README says what that step could not do, which is where the next step starts.

## What was run (21 Oct 2026, macOS arm64)

- `1-numpy-autograd/test.py` (Python 3.12, NumPy 2.5.4, torch 2.14.1): over four runs, each on one random input, the largest gradient difference from PyTorch was 1.1e-8 to 2.9e-7 across `w1`, `b1`, `w2` and `b2`.
- `1-numpy-autograd` notebooks: all four ran top to bottom without errors.
- MNIST, 50 epochs, accuracy on the training set:
  - `experiments/training.py` (string DSL, 784-32-10): 88.0%.
  - `experiments/training_2.py` (`>>` API, with a ReLU layer): 94.3%.
  - The scripts do not evaluate on the test set.
- `2-cpp-cpu/main.cpp`, built with clang++ `-std=c++17` against `scipy_openblas32`: all 12 backward tests passed.
- `2-cpp-cpu/benchmark.py` (784-256-256-10, batch 256, 40 steps): naive matmul 164 ms per step, BLAS 33 ms per step.
- `2-cpp-cpu/training.py` (MNIST, 784-64-10 with no activation, 50 epochs): 88.1% accuracy on the training set, against 88.0% for the linear model in step 1.
- Step 3 needs an NVIDIA GPU and was not built or run, including the fixes listed in its README.
