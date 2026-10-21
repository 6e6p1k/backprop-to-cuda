import numpy as np
from torch import Tensor

class Parameter:

    @staticmethod
    def _reduce_grad(grad: np.ndarray, target_shape: tuple):
        while len(grad.shape) > len(target_shape):
            grad = grad.sum(axis=0)
        for i, (g, t) in enumerate(zip(grad.shape, target_shape)):
            if t == 1 and g != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def __init__(self, data, _prev=(), label='NaN', _grad=None, requires_grad=False, op='') -> None:
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.shape = self.data.shape 
        self.grad = _grad if _grad is not None else np.zeros(self.shape)
        self._prev = set(_prev)
        self._backward = lambda: None
        self._label = label
        self._requires_grad = requires_grad
        self._op = op

    def reset_grad(self):
        self.grad = np.zeros(self.shape)

    def __repr__(self): 
        text = []
        text.extend("Parameter(")
        text.extend(f"shape = {self.shape} | ")
        text.extend(f"size = {self.data.size} | ")
        text.extend(f"label = {self._label}")
        text.extend(")")
        data = f"\ndata:\n{self.data}"
        out = ''.join(text)
        return out

    def __add__(self, other):
        other = other if isinstance(other, Parameter) else Parameter(other)

        out = Parameter(
            data=(self.data + other.data),
            _prev=(self, other),
            op='add'
        )

        def _backward():
            self.grad += self._reduce_grad(out.grad, self.shape)
            other.grad += self._reduce_grad(out.grad, other.shape)

        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Parameter) else Parameter(other)
        try:
            out_data = self.data * other.data
        except Exception as e:
            print("-" * 100)
            print(f"element-wise multiplication failed:")
            print(f"- {e}")
            print(f"  - M1.shape={self.shape}")
            print(f"  - M2.shape={other.shape}")
            print(f"  - M1._label={self._label}, M2._label={other._label}")
            print()
            print(f"M1{self.shape} * M2{other.shape} = ?")
            print("-" * 100)
            raise

        out = Parameter(
            data=(out_data),
            _prev=(self, other),
            op='mul'
        )

        def _backward():
            try:
                self.grad += self._reduce_grad(other.data * out.grad, self.shape)
                other.grad += self._reduce_grad(self.data * out.grad, other.shape)
                    
            except Exception as e:
                print("-" * 100)
                print(f"element-wise backward failed:")
                print(f"- {e}")
                print(f"  - self.shape={self.shape} | {self.data.size=} | self.grad.shape={self.grad.shape}")
                print(f"  - other.shape={other.shape} | {other.data.size=} | other.grad.shape={other.grad.shape}")
                print()
                print(f"  - out.grad.shape={out.grad.shape}")
                print(f"  - self._label={self._label}, other._label={other._label}")
                print()
                print(f"({self.grad.shape=}) += ({other.data.shape=} * {out.grad.shape=}).sum(axis=-1)")
                print(f"({other.grad.shape=}) += ({self.data.shape=} * {out.grad.shape=}).sum(axis=-1)")
                print("-" * 100)
                raise
        
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
            op='matmul'
        )

        def _backward():
            try:
                self.grad += np.einsum('ab,cb->ac', out.grad, other.data)
                other.grad += np.einsum('ab,ac->cb', out.grad, self.data)
            except Exception as e:
                print("-" * 100)
                print(f"matmul backward failed:")
                print(f"- {e}")
                print(f"  - self.shape={self.shape} | self.grad.shape={self.grad.shape}")
                print(f"  - other.shape={other.shape} | other.grad.shape={other.grad.shape}")
                print()
                print(f"  - out.grad.shape={out.grad.shape}")
                print(f"  - self._label={self._label}, other._label={other._label}")
                print()
                print(f"({self.grad.shape=}) += ({np.einsum('ab,cb->ac', out.grad, other.data).shape=})")
                print(f"({other.grad.shape=}) += ({np.einsum('ab,ac->cb', out.grad, self.data).shape=})")
                print("-" * 100)
                raise 

            
        
        out._backward = _backward
        return out
    

    def __pow__(self, other):
        assert isinstance(other, (int, float)) 

        out = Parameter(
            data=(self.data ** other),
            _prev=(self, ),
            label='div_helper' if other == -1 else '',
            op='pow'
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
            op='exp'
        )

        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward
        return out
    
    def sum(self, dim=-1):
        out = Parameter(
            data=self.data.sum(axis=dim, keepdims=True),
            _prev=(self, ),
            label='sum_helper',
            op='sum()'
        )

        def _backward():
            try:
                self.grad += np.broadcast_to(out.grad, self.shape).copy()
            except Exception as e:
                print("-" * 100)
                print(f"sum backward failed:")
                print(f"- {e}")
                print(f"  - self._label={self._label}")
                print(f"  - self.shape={self.shape} | self.grad.shape={self.grad.shape}")
                print(f"  - out.grad.shape={out.grad.shape}")
                print()
                print(f"({self.grad.shape=}) += ({np.broadcast_to(out.grad, self.shape).copy().shape=})")
                print("-" * 100)
                raise
        
        out._backward = _backward
        return out
    
    def mean(self, dim=-1, keepdims=False):
        n = self.data.shape[dim]
        out = Parameter(
            data=np.mean(self.data, axis=dim, keepdims=keepdims),
            _prev=(self, ),
            label='mean_helper',
            op='mean()'
        ) 

        def _backward():
            self.grad += np.broadcast_to(out.grad / n, self.shape).copy()

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
            label='log_helper',
            op='log'
        )

        def _backward():
            self.grad += (1 / (self.data + 1e-8)) * out.grad  # add eps here too

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
    
    def flatten(self):
        out = Parameter(
            data=self.data.reshape(self.shape[0], -1),
            _prev=(self,),
            label='flatten'
        )
        def _backward():
            self.grad += out.grad.reshape(self.shape)
        out._backward = _backward
        return out
    
    def relu(self):
        out = Parameter(
            data=(self.data * (self.data > 0)),
            _prev=(self,),
            label='relu'
        )
        def _backward():
            self.grad += out.grad * (self.data > 0)
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

# ----------------------------------------------------------------------------------------------------------------------------------

import re
from functools import partial

def Linear(x, params):
    w, b = params
    out = x @ w + b
    return out

def make_linear(args):
    in_features, out_features = args
    limit = np.sqrt(6 / (in_features + out_features))
    w = Parameter(np.random.uniform(-limit, limit, (in_features, out_features)), requires_grad=True)
    b = Parameter(np.zeros((1, out_features,)), requires_grad=True)
    return [w, b], Linear

def Softmax(args):
    dim = args[0]

    def sm(x, dim):
        shifted = x - Parameter(x.data.max(axis=dim, keepdims=True))
        exp_x = shifted.exp()
        sum_ = exp_x.sum(dim)
        return exp_x / sum_
    
    return [], partial(sm, dim=dim)

def Flatten(args):
    _ = args
    def flatten(x):
        return x.flatten()

    return [], flatten

LAYERS_AVAILABLE = {
    'Linear' : make_linear,
    'Softmax' : Softmax,
    'Flatten' : Flatten,
}


def extract_between_parentheses(text):
    pattern = r'\((.*)\)'
    match = re.search(pattern, text)
    
    if match:
        content = match.group(1)
        # ints and floats
        numbers = re.findall(r'-?\d+\.?\d*', content)
        result = []
        for num in numbers:
            if '.' in num:
                result.append(float(num))
            else:
                result.append(int(num))
        return result
    return []


def cross_entropy(pred, target, num_classes, batch_size):
    if isinstance(target, np.ndarray):
        target = Parameter(target)
    elif isinstance(target, Tensor):
        target = Parameter(target.clone().detach().numpy())


    def to_ohe_batch(y, num_classes):
        out = np.zeros((len(y.data), num_classes))
        for i, yi in enumerate(y.data.astype(int)):
            out[i, yi] = 1.0
        return Parameter(out, label='ohe_Y')
    
    log_pred = pred.log()
    per_example = -(to_ohe_batch(target, num_classes) * log_pred).sum(dim=-1)  # (128, 1)
    total = per_example.sum(dim=0)   # (1, 1) or (1,)
    return (total * (1 / batch_size)).squeeze()


class Model:
    def __init__(self, architecture, custom_layers=None):
        self.layers = [s.strip() for s in architecture.split('->')]
        self.custom_layers = custom_layers
        self.params = {}
        self.params_list = []
        self.graph = []
        self.layer_names = []
        self.process_layers()

    def process_layers(self):
        for idx in range(len(self.layers)):
            
            layer = self.layers[idx]
            args = extract_between_parentheses(layer)        
            name = layer.split('(')[0]

            params, layer_fn = LAYERS_AVAILABLE[name](args)
            self.params[idx] = params
            self.graph.append(layer_fn)
            self.layer_names.append(name)
            for param in params:
                self.params_list.append(param)

    def __repr__(self): 
        out = []
        out.append("-" * 100)
        out.append(f"Model")
        for i in range(len(self.layers)):
            out.append(f"Layer: {self.layer_names[i]}")
            out.append(f"Parameters: {self.params[i]}")
        out.append("-" * 100)
        return '\n'.join(out)
    
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = Parameter(x)
        elif isinstance(x, Tensor):
            x = Parameter(x.clone().detach().numpy())

        for i in range(len(self.graph)):
            params = self.params[i]


            if params:
                x = self.graph[i](x, params)
            else:
                x = self.graph[i](x)
        return x 
    
    def step(self, optimizer='SGD', lr=1e-3):
        match optimizer:
            case 'SGD':
                for param in self.params_list:
                    if param._requires_grad == True:
                        param.data -= lr * param.grad
            case _:
                raise ValueError(f"Wrong choice! {optimizer=}")

    def zero_grad(self):
        for param in self.params_list:
            if param._requires_grad == True:
                    param.reset_grad()