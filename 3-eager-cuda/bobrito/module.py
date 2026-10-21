from .tensor import Tensor
from .bobr import bobr as rr
from typing import List, Optional
from functools import partial 


def xavier_uniform(shape):
    limit = (6 / (shape[0] + shape[1])) ** (1/2)
    result = rr.uniform(-limit, limit, shape)
    return result

def zeros(shape):
    return rr.zeros(shape)


class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)
    
    def __rshift__(self, other):
        if isinstance(other, Sequential):
            return Sequential(*self.layers, *other.layers)
        return Sequential(*self.layers, other)
    
    def __rrshift__(self, other):
        if isinstance(other, Sequential):
            return Sequential(*other.layers, *self.layers)
        return Sequential(other, *self.layers)
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return ' >> '.join(l._repr() for l in self.layers)
    

class Layer:
    def __init__(self, forward=None, params=None):
        self._forward = forward
        if params is None:
            self._params = []
        else:
            self._params = [
                rr.tsr(init_fn(shape), requires_grad=True) 
                for shape, init_fn in params
                ]

    def __call__(self, x):
        if self._forward is None:
            raise NotImplementedError(f"{self.__class__.__name__} must specify forward")
        if self._params:
            return self._forward(x, *self._params)
        return self._forward(x)

    def __rshift__(self, other):
        return Sequential(self, other)
    
    def parameters(self):
        return self._params
    
    def _repr(self):
        return self.__repr__()
    

class Linear(Layer):
    def __init__(self, in_features, out_features):
        super().__init__(
            forward=lambda x, w, b: x @ w + b,
            params=[
                ((in_features, out_features), xavier_uniform),
                ((1, out_features),           zeros),
            ]
        )
        self.in_features = in_features
        self.out_features = out_features

    def __repr__(self):
        return f"Linear({self.in_features}, {self.out_features})"
    

class LinearBlas(Layer):
    def __init__(self, in_features, out_features):
        super().__init__(
            forward=lambda x, w, b: x.matmul_blas(w) + b,
            params=[
                ((in_features, out_features), xavier_uniform),
                ((1, out_features),           zeros),
            ]
        )
        self.in_features = in_features
        self.out_features = out_features

    def __repr__(self):
        return f"LinearBlas({self.in_features}, {self.out_features})"


class Flatten(Layer):
    def __init__(self):
        super().__init__(forward=lambda x: x.flatten())

    def __repr__(self):
        return f"Flatten()"
    

class Softmax(Layer):

    @staticmethod
    def softmax(x, dim):
        shifted = x - rr.tsr(x.max(dim=dim, keepdims=True))
        exp_x = shifted.exp()
        sum_ = exp_x.sum(dim, keepdims=True)
        return exp_x / sum_ 

    def __init__(self, dim):
        self.dim = dim
        super().__init__(
            forward=partial(self.softmax, dim=dim),
            params=None
        )

    def __repr__(self):
        return f"Softmax({self.dim})"
    

class Loss:
    def __init__(self, loss_fn=None):
        self._loss_fn = loss_fn

    def __call__(self, preds, y):
        if self._loss_fn is None:
            raise NotImplementedError(f"{self.__class__.__name__} must specify loss_fn")

        try:
            loss = self._loss_fn(preds, y)
            if not(isinstance(loss, Tensor)):
                print(f"Invalid loss_fn, should return Tensor, got {type(loss)}.")
                raise ValueError 
            
        except Exception as e:
            print(f"Loss function is invalid: {e}")
            raise ValueError

        return loss.mean(dim=0)
    
    def _repr(self):
        return self.__repr__()
    

class CrossEntropy(Loss):
    @staticmethod
    def cross_entropy_(pred, ohe_target):
        log_pred = pred.log()
        loss = -(ohe_target * log_pred)
        return loss.sum(dim=-1)

    def __init__(self):
        super().__init__(
            loss_fn = self.cross_entropy_
        )

    def __repr__(self):
        return f"Loss: CrossEntropy"
    

class Optimizer:
    def __init__(self, optim_fn=None):
        self._optim_fn = optim_fn

    def step(self, params, lr):
        if self._optim_fn is None:
            raise NotImplementedError(f"{self.__class__.__name__} must specify _optim_fn")

        try:
            self._optim_fn(params, lr) 
            
        except Exception as e:
            print(f"Optimizer function is invalid: {e}")
            raise ValueError

    def _repr(self):
        return self.__repr__()
    

class SGD(Optimizer):
    @staticmethod
    def sgd(params, lr):
        for param in params:
            if param.requires_grad:
                param._sgd_update(lr)

    def __init__(self):
        super().__init__(
            optim_fn = self.sgd
        )

    def __repr__(self):
        return f"Optimizer: SGD"


class Model(Layer):
    def __init__(
            self,
            architecture: Sequential,
            optimizer: Optional[Optimizer] = None,
            loss: Optional[Loss] = None
            ):
        self.net = architecture
        self.optim = optimizer
        self._loss = loss
    
    def __call__(self, x):
        x = rr.tsr(x)
        preds = self.net(x)
        return preds

    def step(self, x_train, y_train, learning_rate=1e-3):
        x = rr.tsr(x_train)
        y_train = rr.tsr(y_train)
        preds = self.net(x)
        loss = self.compute_loss(preds, y_train)

        if self.optim is None:
            print(f"Optimizer is not specified.")
            raise Exception
        
        self.reset_grads()
        loss.backward()
        self.optim.step(self.parameters(), learning_rate)
        return preds, loss

    def reset_grads(self):
        for param in self.parameters():
            if param.requires_grad == True:
                param.reset_grad()
        
    def compute_loss(self, preds, true_values):
        if true_values is None:
            print(f"Y data is not specified.")
            raise Exception
        
        if self._loss is None:
            print(f"Loss function is not specified.")
            raise Exception
        
        loss = self._loss(preds, true_values)
        return loss

    def parameters(self):
        return self.net.parameters()
    
    def __repr__(self):
        return f"Model(\n  {self.net}\n)"
    
    def _repr(self):
        return f"Model( {self.net} )"
    
    def overview(self, show_parameters=False, fold=False):
        out = []
        out.append("*" * 100)
        out.append(f"Architecture: ")

        def _add_info(
            info_lst: list,
            num: int,
            layer: Layer,
            ) -> List[str]:

            if show_parameters:
                params = layer.parameters()
                if params:
                    info_lst.append(f"{num}. {str(layer)} | parameters: ")
                    for idx, param in enumerate(params):
                        info_lst.append(f"\t{idx + 1}) {str(param)}")
                else:
                    info_lst.append(f"{num}. {str(layer)} | [no trainable parameters]")
            else:
                info_lst.append(f"{num}. {str(layer)}")

            return info_lst

        def _iterate_model(model, _count=1):
            for layer in model.net.layers:
                if fold:
                    _add_info(out, _count, layer)
                    continue

                if isinstance(layer, Model):
                    _iterate_model(layer, _count)
                else:
                    _add_info(out, _count, layer)
                _count += 1
        
        _iterate_model(self)
        out.append("*" * 100)
        print('\n'.join(out))

        

