# 4. Bobrito

The framework itself lives in its own repository: [6e6p1k/bobrito](https://github.com/6e6p1k/bobrito).

- Lazy: ops build a graph, and nothing runs until `eval()`.
- The backward pass is emitted as nodes of the same graph and runs on the GPU.
- It has a pooled GPU allocator, CUDA kernels, cuBLAS and CUB, and pybind11 bindings. It trains an MNIST MLP from Python.
