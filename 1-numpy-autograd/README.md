# 1. Autograd in NumPy

- `theory/`: `gradient.ipynb` covers the chain rule with proofs, `main.ipynb` holds the first notes, and `micrograd.ipynb` follows Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) video (unfinished). The topological sort in `Parameter.backward()` comes from that video.
- `parameter.py`: autograd over NumPy arrays, with broadcast-aware gradients, matmul, softmax and cross-entropy.
- `test.py`: checks the gradients of a 784-64-10 network against PyTorch, using an earlier copy of `Parameter` written inline. `grad_version.ipynb` runs the same check against `parameter.py`, with the helpers from `test_tools.py`.
- `experiments/`: two ways to define a model, each trained on MNIST.
  - A string DSL, `Model('Flatten() -> Linear(784, 32) -> Linear(32, 10) -> Softmax(-1)')` from `parameter.py`, in `training.py`.
  - Layers composed with `>>`, `Flatten() >> custom_layer >> Linear(64, 10) >> Softmax(-1)` from `architecture.py`, in `training_2.py`.

The experiments import `parameter` from this folder, so run them from here:

```
PYTHONPATH=. python experiments/training_2.py
```

## What this step couldn't do

- Every node of the graph is a Python object holding a NumPy array, and the backward pass is a Python loop over them.
- `**` only takes a constant exponent, so there is no gradient with respect to the exponent.
- There are no tests with fixed expected values, only comparisons against PyTorch on random inputs.

Step 2 rewrites the engine in C++.
