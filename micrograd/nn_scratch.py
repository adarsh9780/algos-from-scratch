import math
import random
from collections.abc import Iterable


class Value:
    def __init__(
        self,
        data: float,
        label: str | None = None,
        _op: str | None = None,
        _children: Iterable = (),
        grad: float = 0.0,
    ) -> None:
        self.data = data
        self.label = label
        self.grad = grad
        self._op = _op
        self._prev = tuple(_children)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, label={self.label})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _op="+", _children=(self, other))

        def _backward():
            """
            += matters because if one of the inputs is used in multiple
            branches, not using += or using just = will wipe out the
            gradient calculated by the previous steps. therefore, we
            use +=, although it adds another bug if we aren't careful.
            during traning loop, we need to make sure that we zero
            gradients before each run otherwise gradients will accumulate
            from previous runs.
            best example to understand it b = a + a, now backward from b
            till both branches in a.

            step 1: update the gradient of b = 1.0
            step 2: update the gradient 1st branch a
                ðb/ða = out.grad * 1.0 = 1
            step 3: update the gradient of 2nd branch a
            at this point, we already have the gradient calculated from branch 1 which is 1
            ðb/ða = out.grad * 1.0 = 1 => if we do this, it will over write. therefore we have to +=
            and that's why before running the backwards second time, we need to make these gradient zero.
            """
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _op="*", _children=(self, other))

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * (other**-1)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), (
            "only supporting int/float values for power"
        )
        out = Value(self.data**other, _op="**", _children=(self, other))

        def _backward():
            self.grad += out.grad * (other.data * self.data ** (other.data - 1))

        out._backward = _backward

        return out

    def exp(self):
        out = Value(math.exp(self.data), _op="exp", _children=(self,))

        def _backward():
            self.grad += out.grad * out.data

        out._backward = _backward

        return out

    def tanh(self):
        e = (2 * self).exp()
        return (e - 1) / (e + 1)

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
    def __init__(self, nin) -> None:

        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, inputs: list[float]):
        act = [ip * w for ip, w in zip(inputs, self.weights)]
        out = sum(act, self.bias)
        return out.tanh()

    def parameters(self):
        return self.weights + [self.bias]


class Layer:
    def __init__(self, nin: int, nout: int):
        self.layer = [Neuron(nin) for _ in range(nout)]

    def __call__(self, inputs: list[float]):
        outs = [neuron(inputs) for neuron in self.layer]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        params = []
        for neuron in self.layer:
            params.extend(neuron.parameters())

        return params


class MLP:
    def __init__(self, nin: int, nouts: list[int]):
        # Notice nouts is list of integer
        # each number tells us how many output in each layer we want
        sz = [nin] + nouts
        lshape = [(sz[i], sz[i + 1]) for i in range(len(sz) - 1)]
        self.layers = [Layer(i[0], i[1]) for i in lshape]

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
        for p in self.parameters():
            p.grad = 0.0


if __name__ == "__main__":
    # a = Value(3.0, label="a")
    # b = Value(2.0, label="b")
    # c = a + b
    # c.label = "c"
    # d = Value(36.0, label="d")

    # e = d * c
    # e.label = "e"
    # draw_dot(e)

    # a = Value(2.0, label="a")
    # b = a + a
    # draw_dot(b)

    # ðb/ða = ða/ða + ða/ða = 2 * ða/ða = 2

    # manually update the gradients
    # b.backwards()

    # x = Value(0.8814, label="x")
    # y = x.tanh()
    # y.backward()
    # print(f"y: {y.data:.4f}")  # Should be ~0.7071
    # print(f"x.grad: {x.grad:.4f}")  # Should be ~0.5000

    # MLP test
    mlp = MLP(3, [4, 4, 1])

    # 4 training examples, each with 3 features
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]

    # The desired targets (labels)
    ys = [1.0, -1.0, -1.0, 1.0]

    def mse(ypred, y):
        return (ypred - y) ** 2

    epochs = 20
    learning_rate = 0.05
    for i in range(epochs):
        ypreds = [mlp(x) for x in xs]
        error = sum([mse(ypred, y) for ypred, y in zip(ypreds, ys)])

        mlp.zero_grad()
        error.backward()

        parameters = mlp.parameters()

        for p in parameters:
            p.data -= learning_rate * p.grad

        print(f"Epoch: {i:2d} -> Loss: {error.data:.4f}")
