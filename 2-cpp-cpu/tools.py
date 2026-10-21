import os, importlib.util
# windows: dll dir for scipy_openblas32
_spec = importlib.util.find_spec("scipy_openblas32")
if _spec:
    _dll_dir = os.path.join(os.path.dirname(_spec.origin), "lib")
    if os.path.isdir(_dll_dir) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_dll_dir)
import tensor as _C
import math
import random
import numpy as np
import torch
from typing import Optional, Tuple, List

class Tensor:
    def __init__(self, values, shape, requires_grad=False):
        if isinstance(values, _C.Tensor):
            self._t = values
        else:
            if not(isinstance(values, list)):
                raise ValueError(f"Values should have type list, received {type(values)}.")
            if not(isinstance(shape, list)):
                raise ValueError(f"Shape should have type list, received {type(shape)}.")

            self._t = _C.Tensor(values, shape, requires_grad)

    # properties
    @property
    def values(self):
        return self._t.values

    @property
    def shape(self):
        return list(self._t.shape)

    @property
    def grad(self):
        return self._t.grad

    @property
    def requires_grad(self):
        return self._t.requires_grad
    
    @property
    def size(self):
        return math.prod(self.shape)
    
    @property
    def scalar(self):
        if self.size != 1:
            raise ValueError("Tensor is not a scalar.")
        return self._t.values[0]

    # ops
    def __add__(self, other):
        return Tensor(self._t.add(other._t), [])
    
    def __mul__(self, other):
        if isinstance(other, (float, int)):
            other = Tensor([other], [])
        return Tensor(self._t.mul(other._t), [])
    
    def __pow__(self, n):
        assert isinstance(n, (int, float))
        return Tensor(self._t.pow(n), [])
    
    def __sub__(self, other):
        return self + other*-1

    def __truediv__(self, other):
        return self * (other**-1)
    
    def __matmul__(self, other):
        return Tensor(self._t.matmul(other._t), [])

    def matmul_blas(self, other):
        return Tensor(self._t.matmul_blas(other._t), [])
    
    def __neg__(self):
        return self*-1

    def __eq__(self, other):
        return Tensor(self._t.eq(other._t), []) 

    def sum(self, dim=-1, keepdims=False): return Tensor(self._t.sum(dim, keepdims), [])
    def mean(self, dim=-1, keepdims=False): return Tensor(self._t.mean(dim, keepdims), [])
    def max(self, dim=-1, keepdims=False): return Tensor(self._t.max(dim, keepdims), [])
    def argmax(self, dim): return Tensor(self._t.argmax(dim), [])
    def log(self): return Tensor(self._t.log(), [])
    def exp(self): return Tensor(self._t.exp(), [])
    def reshape(self, shape): return Tensor(self._t.reshape(shape), [])
    def transpose(self): return Tensor(self._t.transpose(), [])
    def squeeze(self, dim): return Tensor(self._t.squeeze(dim), [])
    def unsqueeze(self, dim): return Tensor(self._t.unsqueeze(dim), [])
    def flatten(self): return Tensor(self._t.flatten(1, -1), [])
    def reset_grad(self): self._t.reset_grad()
    def update(self, lr=1e-3): self._t.update(lr)
    def backward(self): self._t.backward()

    # repr
    def __repr__(self):
        def fmt_val(v):
            return f"{v:8.4f}"

        def nested(data, shape, offset=0):
            if len(shape) == 1:
                row = ", ".join(fmt_val(data[offset + i]) for i in range(shape[0]))
                return f"[{row}]", offset + shape[0]
            parts = []
            for i in range(shape[0]):
                s, offset = nested(data, shape[1:], offset)
                parts.append(s)
            sep = ",\n" + " " * (len(shape) - 1)
            return "[" + sep.join(parts) + "]", offset

        shape = self.shape
        values = self.values
        if not shape:
            return f"Tensor({fmt_val(values[0])})"

        inner, _ = nested(values, shape)
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({inner}{grad_str})"
    
    def numpy(self):
        # convert to numpy array
        return np.array(self.values, dtype=np.float32).reshape(self.shape)

    

class rr:
    @staticmethod
    def zeros(shape):
        size = math.prod(shape)
        return Tensor([0.0] * size, list(shape))

    @staticmethod
    def ones(shape):
        size = math.prod(shape)
        return Tensor([1.0] * size, list(shape))

    @staticmethod
    def randn(shape):
        size = math.prod(shape)
        return Tensor([random.gauss(0, 1) for _ in range(size)], list(shape))

    @staticmethod
    def arange(n):
        return Tensor(list(range(n)), [n])
    
    @staticmethod
    def uniform(low, high, shape):
        size = math.prod(shape)
        return Tensor([random.uniform(low, high) for _ in range(size)], list(shape))
    
    @staticmethod
    def from_numpy(arr):
        arr = arr.astype(np.float32)
        return Tensor(arr.flatten().tolist(), list(arr.shape))
    
    @staticmethod
    def tsr(x, requires_grad=False):
        if isinstance(x, list):
            return Tensor(x, [len(x)])
        if isinstance(x, Tensor):
            if requires_grad and not x.requires_grad:
                return Tensor(x.values, x.shape, True)
            return x
        if isinstance(x, np.ndarray):
            return Tensor(x.flatten().tolist(), list(x.shape), requires_grad)
        try:
            if isinstance(x, torch.Tensor):
                arr = x.detach().numpy().astype(np.float32)
                return Tensor(arr.flatten().tolist(), list(arr.shape), requires_grad)
        except ImportError:
            pass
        raise TypeError(f"Cannot convert {type(x)} to Tensor.")
    
    @staticmethod
    def argmax(
        x: Tensor,
        dim: int = -1,
        ):
        return x.argmax(dim)
    
    @staticmethod
    def mean(
        x,
        dim: int = -1,
        keepdims=False
        ):
        if isinstance(x, Tensor):
            return x._t.mean(dim, keepdims)
        elif isinstance(x, list):
            return Tensor(x, [len(x)]).mean(dim, keepdims)
        else:
            raise ValueError(f"Use Tensor.mean().")
        

        
        

