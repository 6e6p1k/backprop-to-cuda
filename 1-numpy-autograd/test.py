import numpy as np
import torch
from contextlib import contextmanager
import time

@contextmanager
def timing(label=''):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{label+': ' if label else ''}{end-start:.8f}s")

class Parameter:
    def __init__(self, data, _prev=(), label='', _grad=None) -> None:
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.shape = self.data.shape 
        self.grad = _grad if _grad is not None else np.zeros(self.shape)
        self._prev = set(_prev)
        self._backward = lambda: None
        self._label = label

    def __repr__(self):
        return f"[shape = {self.shape}\nsize = {self.data.size}\ngrad.shape = {self.grad.shape}\nlabel = {self._label}\ndata: {self.data}]"
        

    def __add__(self, other):
        assert self.shape == other.shape
        other = other if isinstance(other, Parameter) else Parameter(other)

        out = Parameter(
            data=(self.data + other.data),
            _prev=(self, other),
        )

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Parameter) else Parameter(other)
        assert self.shape == other.shape or self.data.size == 1 or other.data.size == 1

        out = Parameter(
            data=(self.data * other.data),
            _prev=(self, other),
        )

        def _backward():
            self.grad += (other.data * out.grad)
            if other.grad.size == 1:
                other.grad += (self.data * out.grad).sum()    
            else:
                other.grad += (self.data * out.grad)

        
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __matmul__(self, other):
        other = other if isinstance(other, Parameter) else Parameter(other)
        assert self.shape[-1] == other.shape[0]

        out = Parameter(
            data=(self.data @ other.data),
            _prev=(self, other),
        )

        def _backward():
            self.grad += (other.data @ out.grad.transpose()).reshape(self.grad.shape)
            other.grad += (self.data.transpose() @ out.grad).reshape(other.grad.shape)
        
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)) 

        out = Parameter(
            data=(self.data ** other),
            _prev=(self, ),
            label='div_helper' if other == -1 else '',
        )

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def exp(self):

        out = Parameter(
            data=(np.exp(self.data)),
            _prev=(self, ),
        )

        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward
        return out
    
    def sum(self, dim=-1):
        out = Parameter(
            data=self.data.sum(axis=dim),
            _prev=(self, ),
            _grad=np.zeros(self.shape)
        )

        def _backward():
            self.grad += np.ones(self.shape) * out.grad
        
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + other * -1

    def log(self):
        out = Parameter(
            data=np.log(self.data + 1e-8),
            _prev=(self,),
        )
        
        def _backward():
            self.grad += (1 / self.data) * out.grad
            
        out._backward = _backward
        return out

    def squeeze(self):
        out = Parameter(
            data = self.data.squeeze(),
            _prev=(self, )
        )

        def _backward():
            self.grad += out.grad.reshape(self.shape)
        
        out._backward = _backward
        return out
        
    def backward(self):
        self.grad = np.ones_like(self.data)
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        for node in reversed(topo):
            node._backward()
    
# PARAMETERS -------------------------------------------------------------------------------


"""TORCH"""
# Linear 1 (784, 64) (assuming batch_size=1)
w1 = torch.empty(784, 64); w1.requires_grad = True
b1 = torch.rand(1, 64); b1.requires_grad = True

# Linear 2 (64, 10)
w2 = torch.empty(64, 10); w2.requires_grad = True
b2 = torch.rand(1, 10); b2.requires_grad = True

# softmax
sm = torch.nn.Softmax(dim=-1)

# Loss function
loss_fn = torch.nn.NLLLoss()

torch.nn.init.xavier_uniform_(w1)
torch.nn.init.xavier_uniform_(w2)

x = torch.rand(1, 784)
target = torch.tensor([4])


"""CUSTOM"""
w1_ = Parameter(w1.detach().numpy(), label='w1')
b1_ = Parameter(b1.detach().numpy(), label='b1')
w2_ = Parameter(w2.detach().numpy(), label='w2')
b2_ = Parameter(b2.detach().numpy(), label='b2')
num_classes = 10

def sm_(x: Parameter, dim=-1) -> Parameter:
    exp_x = x.exp();        exp_x._label='exp'
    sum_ = exp_x.sum(dim);  sum_._label='sum'
    out = exp_x / sum_
    return out

def to_ohe(y, num_classes):
    out = np.zeros(num_classes)
    out[y.data] = 1.0
    return Parameter(out)

def cross_entropy(pred, target):
    # pred: softmax output, shape (num_classes,)
    # target: one-hot, shape (num_classes,)
    log_pred = pred.log()
    return -(to_ohe(target, num_classes) * log_pred).sum()


x_ = Parameter(x.clone().numpy(), label='x')
target_ = Parameter(target.clone().numpy(), label='targets')


# CALCULATIONS -------------------------------------------------------------------------------


"""TORCH"""
with timing('TORCH'):
    o_1 = x @ w1 + b1
    o_2 = o_1 @ w2 + b2
    
    o_sm = torch.log(sm(o_2))
    o_l = loss_fn(o_sm, target)
    
    o_l.backward()


"""CUSTOM"""
with timing('PARAMETER CLASS'):
    o_1_ = x_ @ w1_;      o_1_._label = 'o_1_'
    o_2_ = o_1_ + b1_;    o_2_._label = 'o_2_'
    o_3_ = o_2_ @ w2_;    o_3_._label = 'o_3_'
    o_4_ = o_3_ + b2_;    o_4_._label = 'o_4_'
    
    o_sm_ = sm_(o_4_).squeeze();       o_sm_._label = 'o_sm'
    o_l_ = cross_entropy(o_sm_, target_);     o_l_._label= 'o_l'
    
    o_l_.backward()


# OUTPUT COMPARISON -------------------------------------------------------------------------------
def compare_grads(custom_param, torch_param, name=''):
    a = custom_param.grad
    b = torch_param.grad.numpy()
    max_diff = np.abs(a - b).max()
    close = np.allclose(a, b, rtol=1e-3, atol=1e-5)
    print(f"{name+': ' if name else ''}max_diff={max_diff:.2e} {'✓' if close else '✗'}")

compare_grads(w1_, w1, 'w1')
compare_grads(w2_, w2, 'w2')
compare_grads(b1_, b1, 'b1')
compare_grads(b2_, b2, 'b2')