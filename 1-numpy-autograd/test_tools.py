from contextlib import contextmanager
import time
import numpy as np
from torch import Tensor
from parameter import Parameter

@contextmanager
def timing(label=''):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{label+': ' if label else ''}{end-start:.8f}s")


def compare_grads(custom_param, torch_param, name=''):
    a = custom_param.grad
    b = torch_param.grad.numpy()
    max_diff = np.abs(a - b).max()
    close = np.allclose(a, b, rtol=1e-3, atol=1e-5)
    print(f"{name+': ' if name else ''}max_diff={max_diff:.2e} {'✓' if close else '✗'}")


def tensor_to_parameter(tensor: Tensor, **kwargs):
    return Parameter(tensor.detach().numpy(), **kwargs)
