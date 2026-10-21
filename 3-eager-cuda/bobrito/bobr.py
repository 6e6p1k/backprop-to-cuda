from .tensor import Tensor
from . import bobrito_cpp as _C
import math
import random
import numpy as np
from torch import Tensor as __tensor
from typing import List, Tuple
import torch

def parse_args(args, kwargs={}):
    # Normalize shape: zeros(3,4) or zeros((3,4))
    if len(args) >= 1 and isinstance(args[0], (tuple, list)):
        shape = tuple(args[0])
        rest = args[1:]
    else:
        # consume ints from the front as shape
        i = 0
        while i < len(args) and isinstance(args[i], int):
            i += 1
        shape = tuple(args[:i])
        rest = args[i:]

    # positional fallback: zeros(shape, requires_grad, device)
    requires_grad = rest[0] if len(rest) > 0 else kwargs.get('requires_grad', False)
    device        = rest[1] if len(rest) > 1 else kwargs.get('device', 'cpu')

    return shape, requires_grad, device

class bobr:
    @staticmethod
    def zeros(*args, **kwargs):
        shape, requires_grad, device = parse_args(args, kwargs)
        size = math.prod(shape)
        return Tensor([0.0] * size, list(shape), requires_grad=requires_grad, device=device)

    @staticmethod
    def ones(*args, **kwargs):
        shape, requires_grad, device = parse_args(args, kwargs)
        size = math.prod(shape)
        return Tensor([1.0] * size, list(shape), requires_grad=requires_grad, device=device)

    @staticmethod
    def randn(*args, **kwargs):
        shape, requires_grad, device = parse_args(args, kwargs)
        size = math.prod(shape)
        return Tensor([random.gauss(0, 1) for _ in range(size)], list(shape), requires_grad=requires_grad, device=device)

    @staticmethod
    def arange(*args, requires_grad=False, device='cpu'):
        values = [float(x) for x in range(*args)]
        return Tensor(values, [len(values)], requires_grad=requires_grad, device=device)
    
    @staticmethod
    def uniform(low, high, shape):
        size = math.prod(shape)
        return Tensor([random.uniform(low, high) for _ in range(size)], list(shape))
    
    @staticmethod
    def from_numpy(arr: np.ndarray) -> Tensor:
        return Tensor._from_numpy(arr)
    
    @staticmethod
    def from_torch(tsr):
        return Tensor._from_torch(tsr)
    
    @staticmethod
    def tsr(x, requires_grad=None, device=None):
        try:
            if isinstance(x, list):
                if device is None:
                    device = "cpu"
                if requires_grad is None:
                    requires_grad = False
                return Tensor(x, [len(x)], device=device, requires_grad=requires_grad)
            if isinstance(x, Tensor):
                if device is None:
                    device = x.device
                if requires_grad is None:
                    requires_grad = False
                return Tensor(x.values, shape=x.shape, device=device, requires_grad=requires_grad)
            if isinstance(x, np.ndarray):
                return Tensor._from_numpy(x)
            if isinstance(x, torch.Tensor):
                return Tensor._from_torch(x)
        except:
            raise TypeError(f"Cannot convert {type(x)} to Tensor.")
    
    @staticmethod
    def argmax(
        x: Tensor,
        dim: int = -1,
        ):
        return x.argmax(dim)
    
    @staticmethod
    def mean(x, dim: int = -1, keepdims=False):
        
        if isinstance(x, list):
            x = Tensor(x, [len(x)])
            return x.mean()
        if isinstance(x, Tensor):
            return x.mean()    
        if isinstance(x, np.ndarray):
            return x.mean()
        if isinstance(x, torch.Tensor):
            return x.mean()
        
        raise TypeError(f"Cannot compute mean of type {type(x)}.")

    @staticmethod
    def to_cuda(x: "Tensor") -> "Tensor":
        return x.to_cuda()

    @staticmethod
    def to_cpu(x: "Tensor") -> "Tensor":
        return x.to_cpu()
        
