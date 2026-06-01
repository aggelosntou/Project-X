"""
engine.py — Scalar-valued automatic differentiation engine.

Automatic differentiation (autograd) is the core of every modern deep learning
framework.  The key idea is to record every arithmetic operation in a directed
acyclic graph (DAG) as it happens, then replay it in reverse to compute
gradients via the chain rule.

This is NOT symbolic differentiation (like SymPy) — we compute numerical values
at every step.  And it is NOT finite differences — we get exact gradients.

Reference: Andrej Karpathy's micrograd, extended with more operations.
"""

import math


class Value:
    """
    A node in the computation graph.

    Every Value stores:
      - data:      the forward-pass scalar value (float)
      - grad:      dL/d(self) — gradient of the loss w.r.t. this value (starts at 0)
      - _backward: closure that computes dL/d(parents) from dL/d(self)
      - _prev:     set of parent Values (inputs that created this Value)
      - _op:       string describing which operation created this node (for debugging)

    The computation graph is a DAG: edges point from children to parents.
    backward() traverses this graph from output (loss) to inputs (parameters).
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0          # dL/d(self), starts at zero
        self._backward = lambda: None  # filled in by each operation
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f}, op='{self._op}')"

    # ------------------------------------------------------------------ #
    #  Forward operations                                                  #
    # ------------------------------------------------------------------ #

    def __add__(self, other):
        """
        z = a + b
        Chain rule:
            dL/da = dL/dz * dz/da = dL/dz * 1 = dL/dz
            dL/db = dL/dz * dz/db = dL/dz * 1 = dL/dz
        Addition distributes gradients equally to both inputs — it's a "fan-out".
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # dL/da += dL/dz ('+=' because a node may be used multiple times)
            self.grad  += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        """
        z = a * b
        Chain rule:
            dL/da = dL/dz * b    (the other factor becomes the gradient multiplier)
            dL/db = dL/dz * a
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        """
        z = a^n  (n must be a Python int or float, not a Value)
        Chain rule:
            dL/da = dL/dz * n * a^(n-1)    (power rule)
        """
        assert isinstance(exponent, (int, float)), "exponent must be a scalar"
        out = Value(self.data ** exponent, (self,), f'**{exponent}')

        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        # a / b = a * b^(-1)
        return self * (other ** -1)

    def __radd__(self, other):  return Value(other) + self
    def __rmul__(self, other):  return Value(other) * self
    def __rsub__(self, other):  return Value(other) - self
    def __rtruediv__(self, other): return Value(other) / self

    # ------------------------------------------------------------------ #
    #  Activation functions                                                #
    # ------------------------------------------------------------------ #

    def relu(self):
        """
        z = max(0, a)
        Chain rule: dL/da = dL/dz if a > 0, else 0
        """
        out = Value(max(0.0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        """
        z = tanh(a) = (e^a - e^{-a}) / (e^a + e^{-a})
        Chain rule: dL/da = dL/dz * (1 - tanh^2(a))
        Note: 1 - tanh^2(a) = sech^2(a), the "squashing" factor.
        When |a| is large, tanh saturates → gradient ≈ 0 → vanishing gradient.
        """
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        """
        z = sigma(a) = 1 / (1 + e^{-a})
        Chain rule: dL/da = dL/dz * sigma(a) * (1 - sigma(a))
        Elegant: the derivative of sigmoid IS expressible in terms of sigmoid itself.
        """
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        """
        z = e^a
        Chain rule: dL/da = dL/dz * e^a = dL/dz * z
        The exponential function is its own derivative — the magic of e.
        """
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            self.grad += e * out.grad   # e == out.data

        out._backward = _backward
        return out

    def log(self):
        """
        z = ln(a)   (natural log)
        Chain rule: dL/da = dL/dz / a
        Requires a > 0.  log appears in cross-entropy loss.
        """
        assert self.data > 0, f"log of non-positive: {self.data}"
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    #  Backward pass                                                       #
    # ------------------------------------------------------------------ #

    def backward(self):
        """
        Compute gradients of this Value w.r.t. all leaf Values in the graph.

        Algorithm:
          1. Build a topological ordering of the graph (parents before children).
          2. Set self.grad = 1.0  (dL/dL = 1, the seed gradient).
          3. Visit nodes in reverse topological order, calling _backward()
             on each.  This propagates gradients from output to inputs via
             the chain rule.

        Why topological sort?
          We can only compute dL/d(parent) after we know dL/d(child).
          Topological order guarantees every node is processed after all its
          downstream consumers — so its .grad is fully accumulated before
          we call its _backward().
        """
        topo = []
        visited = set()

        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)   # append AFTER processing parents → topo is parents-last

        build_topo(self)

        self.grad = 1.0  # seed: dL/dL = 1
        for node in reversed(topo):  # reverse → process output first, inputs last
            node._backward()

    def zero_grad(self):
        """Set gradient to zero. Call before each backward pass."""
        self.grad = 0.0
