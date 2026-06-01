"""
layers.py — Neural network layers in pure NumPy.

Each layer implements:
  - forward(x): compute output, cache inputs for backward
  - backward(grad_out): compute grad_in, grad_weights, grad_biases
  - update(lr): apply SGD gradient descent

The math here is identical to the C implementation in neural-net-c,
but NumPy makes the matrix operations clear and concise.
"""

import numpy as np


class Linear:
    """
    Fully-connected (dense) layer.

    Forward: z = x @ W + b
      x: (batch, in_features)
      W: (in_features, out_features)
      b: (1, out_features)
      z: (batch, out_features)

    Backward:
      dL/dW = x^T @ grad_out      [shape: (in_features, out_features)]
      dL/db = mean(grad_out, 0)   [shape: (1, out_features)]
      dL/dx = grad_out @ W^T      [shape: (batch, in_features)]

    Derivation of dL/dW:
      L depends on z, z = x @ W + b, so by chain rule:
        dL/dW_{ij} = sum_n dL/dz_{nj} * x_{ni}
      In matrix form: dL/dW = x^T @ dL/dz
    """

    def __init__(self, in_features: int, out_features: int):
        # Xavier uniform initialization: preserves variance across layers.
        # See neural-net-c LEARNINGS.md for the derivation.
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        self.b = np.zeros((1, out_features))

        # Stored for backward pass
        self._x = None
        # Stored gradients (updated by backward, applied by update)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x  # cache for backward
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        grad_out: dL/dz, shape (batch, out_features)
        Returns: dL/dx, shape (batch, in_features)
        """
        self.dW = self._x.T @ grad_out              # (in, batch) @ (batch, out)
        self.db = grad_out.mean(axis=0, keepdims=True)  # average over batch
        return grad_out @ self.W.T                  # propagate gradient backward

    def update(self, lr: float):
        self.W -= lr * self.dW
        self.b -= lr * self.db

    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    """
    Rectified Linear Unit: f(x) = max(0, x)

    Forward: a = max(0, z)
    Backward: dL/dz = dL/da * I[z > 0]   (gate by "was the neuron on?")

    The indicator I[z > 0] is computed from the forward-pass input z.
    """

    def __init__(self):
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = (x > 0)
        return x * self._mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self._mask

    def update(self, lr: float):
        pass  # no parameters

    def parameters(self):
        return []


class Sigmoid:
    """
    Sigmoid: σ(x) = 1 / (1 + e^{-x})

    Backward: dL/dx = dL/dσ * σ(1 - σ)
    Note: we cache σ(x), not x, because the derivative is expressed in terms of σ.
    """

    def __init__(self):
        self._s = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Numerically stable: handle positive and negative x separately
        s = np.where(x >= 0,
                     1 / (1 + np.exp(-x)),
                     np.exp(x) / (1 + np.exp(x)))
        self._s = s
        return s

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self._s * (1 - self._s)

    def update(self, lr: float):
        pass

    def parameters(self):
        return []


class Tanh:
    """
    Hyperbolic tangent: tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})

    Backward: dL/dx = dL/dtanh * (1 - tanh^2(x)) = dL/dtanh * sech^2(x)
    """

    def __init__(self):
        self._t = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._t = np.tanh(x)
        return self._t

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * (1 - self._t ** 2)

    def update(self, lr: float):
        pass

    def parameters(self):
        return []


class Softmax:
    """
    Row-wise numerically-stable softmax.

    Softmax is rarely used as an intermediate activation — it's typically
    applied only at the output layer. In practice, it's fused with the
    cross-entropy loss for numerical stability (see losses.py).

    Forward: p_j = exp(x_j - max_x) / sum_k exp(x_k - max_x)

    Backward (for standalone use): The Jacobian of softmax is
        d softmax_i / d x_j = p_i (delta_{ij} - p_j)
    where delta is the Kronecker delta. In compact form:
        dL/dx = p * (dL/dp - sum(dL/dp * p))
    """

    def __init__(self):
        self._p = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Subtract max along last axis for numerical stability
        x_shifted = x - x.max(axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        self._p = exp_x / exp_x.sum(axis=-1, keepdims=True)
        return self._p

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        # dL/dx_i = p_i * (dL/dp_i - sum_j(dL/dp_j * p_j))
        dot = (grad_out * self._p).sum(axis=-1, keepdims=True)
        return self._p * (grad_out - dot)

    def update(self, lr: float):
        pass

    def parameters(self):
        return []
