import math
import random
from collections.abc import Iterable

# from draw import draw_dot


class Value:
    def __init__(
        self,
        data: float,
        label: str = "",
        _op: str | None = None,
        _children: Iterable = (),
        grad: float = 0.0,
    ):
        self.data = data
        self.label = label
        self.grad = grad
        self._op = _op
        self._prev = set(_children)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, label={self.label})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            other.grad += (
                out.grad
            )  # we want to accumulate the gradients from L till here
            self.grad += out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, _children=(self, other), _op="*")

        def _backward():
            other.grad += out.grad * self.data
            self.grad += out.grad * other.data

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def exp(self):
        out = Value(math.exp(self.data), _op="exp", _children=(self,))

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), (
            "only supporting int/float powers for now"
        )
        out = Value(self.data**other, _children=(self,), _op=f"**{other}")

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        out = self * (other**-1)
        return out

    def __sub__(self, other):
        out = self + (-1 * other)
        return out

    def tanh(self):
        e = (2 * self).exp()
        e.label = "e"
        o = (e - 1) / (e + 1)
        o.label = "tanh"
        return o

    def backward(self):
        self.grad = 1.0
        nodes = traverse(self)
        for n in nodes[::-1]:
            n._backward()


def traverse(node: Value) -> list:
    visited = set[Value]()
    topo = []

    def build(node: Value):
        if node in visited:
            return
        visited.add(node)  # <-- 1. Mark as visited
        for c in node._prev:
            build(c)
        topo.append(node)

    build(node)
    return topo  # <-- 2. Return topo list!


class Neuron:
    def __init__(self, nin):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1, 1), label="bias")

    def __call__(self, inputs):
        act = [i * w for i, w in zip(inputs, self.weights)]
        out = sum(act, self.bias)
        return out.tanh()

    def parameters(self):
        return self.weights + [
            self.bias
        ]  # <-- notice bias in a list, add as an item to the list [w1, w2, b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, inputs):
        outs = [n(inputs) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        # params = []
        # for n in self.neurons:
        #     params.extend(n.parameters())
        # return params
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin: int, nouts: list[int]) -> None:
        # [3, 4, 4, 1]
        self.sz = [nin] + nouts
        self.layers = [
            Layer(self.sz[i], self.sz[i + 1]) for i in range(len(self.sz) - 1)
        ]

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())

        return params

    def zero_grad(self):
        """
        After each time we backpropgate the gradients, most of the time
        we would like to reset the gradient otherwise it would double up
        because of '+=' we have used.

        the idea is that once the weights are updated based on the new gradient,
        we would like to start again from dL/dL = 1 and re do the whole process.

        So, before calling self.backward(), we should call self.zero_grad()
        """
        for p in self.parameters():
            p.grad = 0.0


if __name__ == "__main__":
    # MLP test
    mlp = MLP(3, [4, 4, 1])

    # 4 training examples, each with 3 features
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]

    # The desired targets (labels)
    ys = [1.0, -1.0, -1.0, 1.0]

    def mse(ypred, y):
        return (ypred - y) ** 2

    epochs = 160
    learning_rate = 0.05
    for i in range(epochs):
        # make predictions: forward pass
        pred = [mlp(x) for x in xs]

        # compute the loss
        loss = sum([mse(ypred, y) for ypred, y in zip(pred, ys)])

        # backpropgate
        mlp.zero_grad()
        loss.backward()

        parameters = mlp.parameters()

        for p in parameters:
            p.data -= p.grad * learning_rate

        print(f"i:{i:2d} --> loss={loss.data:.4f}")
