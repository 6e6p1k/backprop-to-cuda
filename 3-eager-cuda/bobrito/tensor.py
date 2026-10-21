import math
import numpy as np
import torch as _torch

# init cuda through torch first so both use the same libcudart
try:
    if _torch.cuda.is_available():
        _torch.cuda.init()
except Exception:
    pass

from . import bobrito_cpp as _C


class Tensor:
    def __init__(self, values, shape, device="cpu", requires_grad=False):
        if isinstance(values, _C.Tensor):
            self._t = values
        else:
            if not(isinstance(values, list)):
                raise ValueError(f"Values should have type list, received {type(values)}.")
            if not(isinstance(shape, list)):
                raise ValueError(f"Shape should have type list, received {type(shape)}.")

            self._t = _C.Tensor(values, shape, device, requires_grad)

    # properties
    @property
    def values(self):
        return self._t._out_full_values()

    @property
    def shape(self):
        return list(self._t.shape)

    @property
    def grad(self):
        return self._t.grad
    
    @property
    def device(self):
        return self._t.device
    
    @device.setter
    def device(self, other):
        _devices = ["cpu", "cuda"]
        if self.requires_grad == True:
            raise SystemError("Cannot change device when requires_grad=True.")
        if other == "cuda":
            self.to_cuda()
        elif other == "cpu":
            self.to_cpu()
        else:
            raise ValueError(f"Expected one of {', '.join(_devices)}, received {other}.")
        

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
        return self.values[0]

    # operators
    def __add__(self, other):
        if isinstance(other, (float, int)):
            other = Tensor([float(other)], [1], device=self.device)
        return Tensor(self._t.add(other._t), [])
    
    def __mul__(self, other):
        if isinstance(other, (float, int)):
            other = Tensor([other], [], device=self.device)
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

    def matmul_fast(self, other):
        return Tensor(self._t.matmul_fast(other._t), [])

    def matmul_blas(self, other):
        return Tensor(self._t.matmul_blas(other._t), [])

    def matmul_cuda(self, other):
        return Tensor(self._t.matmul_cuda(other._t), [])

    def to_cuda(self):
        return Tensor(self._t.to_cuda(), [])

    def to_cpu(self):
        return Tensor(self._t.to_cpu(), [])
    
    def __neg__(self):
        return self*-1

    def __eq__(self, other):
        return Tensor(self._t.eq(other._t), []) 

    def sum(self, dim=-1, keepdims=False): 
        return Tensor(self._t.sum(dim, keepdims), [])
    
    def mean(self, dim=-1, keepdims=False): 
        return Tensor(self._t.mean(dim, keepdims), [])
    
    def max(self, dim=-1, keepdims=False): 
        return Tensor(self._t.max(dim, keepdims), [])

    def argmax(self, dim): 
        return Tensor(self._t.argmax(dim), [])
    
    def log(self): 
        return Tensor(self._t.log(), [])
    
    def exp(self): 
        return Tensor(self._t.exp(), [])
    
    def reshape(self, shape): 
        return Tensor(self._t.reshape(shape), [])
    
    def transpose(self): 
        return Tensor(self._t.transpose(), [])
    
    def squeeze(self, dim): 
        return Tensor(self._t.squeeze(dim), [])
    
    def unsqueeze(self, dim): 
        return Tensor(self._t.unsqueeze(dim), [])
    
    def flatten(self): 
        return Tensor(self._t.flatten(1, -1), [])
    
    def reset_grad(self): 
        self._t.reset_grad()
    
    def _sgd_update(self, lr=1e-3): 
        self._t.sgd_update(lr)
    
    def backward(self): 
        self._t.backward()

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
    
    def numpy(self) -> np.ndarray:
        # convert to numpy array
        return np.array(self.values, dtype=np.float32).reshape(self.shape)
    
    @staticmethod
    def _from_numpy(arr: np.ndarray):
        arr = arr.astype(np.float32)
        return Tensor(arr.flatten().tolist(), list(arr.shape))
    
    def torch(self) -> _torch.Tensor:
        # converts to torch tensor
        return _torch.tensor(self.values, dtype=_torch.float32, device=self.device).reshape(self.shape)
    
    @staticmethod
    def _from_torch(tsr : _torch.Tensor):
        tsr = tsr.type(_torch.float32)
        return Tensor(tsr.flatten().tolist(), list(tsr.shape), requires_grad=tsr.requires_grad, device=tsr.device.type)

