import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        if isinstance(data, (int, float)):
            data = np.array(data, dtype=np.float64)
        elif isinstance(data, list):
            data = np.array(data, dtype=np.float64)
        elif isinstance(data, np.ndarray):
            data = data.astype(np.float64) if data.dtype != np.float64 else data
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        out = Tensor(self.data.T, requires_grad=self.requires_grad, _children=(self,), _op='T')
        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad.T)
        out._backward = _backward
        return out

    def _accumulate_grad(self, grad):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += grad

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad, _children=(self,), _op='reshape')
        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad.reshape(self.data.shape))
        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float64))
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op='+')

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if grad.shape != self.data.shape:
                    ndims_added = grad.ndim - self.data.ndim
                    for _ in range(ndims_added):
                        grad = grad.sum(axis=0)
                    for i, s in enumerate(self.data.shape):
                        if s == 1:
                            grad = grad.sum(axis=i, keepdims=True)
                self._accumulate_grad(grad)
            if other.requires_grad:
                grad = out.grad
                if grad.shape != other.data.shape:
                    ndims_added = grad.ndim - other.data.ndim
                    for _ in range(ndims_added):
                        grad = grad.sum(axis=0)
                    for i, s in enumerate(other.data.shape):
                        if s == 1:
                            grad = grad.sum(axis=i, keepdims=True)
                other._accumulate_grad(grad)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float64))
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op='*')

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(other.data * out.grad)
            if other.requires_grad:
                other._accumulate_grad(self.data * out.grad)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float64))
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op='/')

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad / other.data)
            if other.requires_grad:
                other._accumulate_grad(-self.data * out.grad / (other.data ** 2))
        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float64))
        return other / self

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float64))
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op='@')

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad @ other.data.T)
            if other.requires_grad:
                other._accumulate_grad(self.data.T @ out.grad)
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _children=(self,), _op='sum')

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self._accumulate_grad(np.broadcast_to(grad, self.data.shape).copy())
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n = self.data.size
        elif isinstance(axis, int):
            n = self.data.shape[axis]
        else:
            n = 1
            for a in axis:
                n *= self.data.shape[a]
        return self.sum(axis=axis, keepdims=keepdims) / n

    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad, _children=(self,), _op='exp')

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.data * out.grad)
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad, _children=(self,), _op='log')

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad / self.data)
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad, _children=(self,), _op='relu')

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad * (self.data > 0).astype(np.float64))
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, requires_grad=self.requires_grad, _children=(self,), _op='sigmoid')

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad * s * (1 - s))
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad, _children=(self,), _op='tanh')

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad * (1 - t ** 2))
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exp_vals = np.exp(shifted)
        s = exp_vals / np.sum(exp_vals, axis=axis, keepdims=True)
        out = Tensor(s, requires_grad=self.requires_grad, _children=(self,), _op='softmax')

        def _backward():
            if self.requires_grad:
                ds = s * (out.grad - np.sum(out.grad * s, axis=axis, keepdims=True))
                self._accumulate_grad(ds)
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        self.grad = None

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
