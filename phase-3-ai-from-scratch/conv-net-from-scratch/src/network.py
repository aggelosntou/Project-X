"""
network.py — Sequential container and common layers for building CNNs.

Provides:
  Linear      — fully-connected layer
  ReLU        — rectified linear activation
  Softmax     — output probability distribution
  Sequential  — chains layers; forward() and backward() in order
  BatchNorm2d — batch normalization for conv feature maps (bonus)
"""

import numpy as np


# ── Activation functions ───────────────────────────────────────────────────────

class ReLU:
    """ReLU activation: f(x) = max(0, x)"""

    def __init__(self):
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = (x > 0)
        return x * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self._mask

    def parameters(self):
        return []


class Softmax:
    """
    Softmax activation (numerically stable).
    For output layer — use CrossEntropyLoss which combines log-softmax + NLL
    for better numerical stability. This standalone class is for inspection only.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        shifted = x - x.max(axis=-1, keepdims=True)
        exp_x   = np.exp(shifted)
        probs   = exp_x / exp_x.sum(axis=-1, keepdims=True)
        self._probs = probs
        return probs

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # d(softmax) full Jacobian is complex; use with NLL loss instead
        # For softmax + NLL combined: gradient = probs - one_hot_target
        p = self._probs
        return p * (dout - (dout * p).sum(axis=-1, keepdims=True))

    def parameters(self):
        return []


# ── Linear layer ───────────────────────────────────────────────────────────────

class Linear:
    """
    Fully-connected layer: out = x @ W + b

    Parameters
    ----------
    in_features  : input dimension
    out_features : output dimension

    Initialization: He (Kaiming) for use after ReLU.
      W ~ N(0, sqrt(2 / in_features))
    """

    def __init__(self, in_features: int, out_features: int):
        self.in_features  = in_features
        self.out_features = out_features

        # He initialization (optimal for ReLU activations)
        std = np.sqrt(2.0 / in_features)
        self.W  = np.random.randn(in_features, out_features) * std
        self.b  = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N, in_features) → (N, out_features)"""
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        dout: (N, out_features)
        dW = x.T @ dout,  db = dout.sum(0),  dx = dout @ W.T
        """
        self.dW += self._x.T @ dout
        self.db += dout.sum(axis=0)
        return dout @ self.W.T

    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]

    def zero_grad(self):
        self.dW[:] = 0
        self.db[:] = 0


# ── Sequential container ───────────────────────────────────────────────────────

class Sequential:
    """
    Chain of layers: each layer's output is the next layer's input.

    Usage:
        model = Sequential([
            Conv2d(1, 8, 3), ReLU(), MaxPool2d(2),
            Flatten(),
            Linear(1568, 10),
        ])
        out  = model.forward(x)
        grad = model.backward(d_out)
    """

    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        for p, g in self.parameters():
            g[:] = 0

    def __repr__(self):
        lines = ["Sequential("]
        for i, layer in enumerate(self.layers):
            lines.append(f"  ({i}) {layer.__class__.__name__}")
        lines.append(")")
        return "\n".join(lines)


# ── Cross-entropy loss ─────────────────────────────────────────────────────────

def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray):
    """
    Softmax + NLL loss.

    logits:  (N, num_classes)
    targets: (N,)  integer class indices

    Returns: (scalar loss, gradient w.r.t. logits)

    Gradient: (softmax(logits) - one_hot(targets)) / N
    """
    N = logits.shape[0]
    C = logits.shape[1]

    # Numerically stable log-softmax
    shifted  = logits - logits.max(axis=1, keepdims=True)
    exp_z    = np.exp(shifted)
    softmax  = exp_z / exp_z.sum(axis=1, keepdims=True)
    log_prob = np.log(softmax + 1e-9)

    # NLL loss
    loss = -log_prob[np.arange(N), targets].mean()

    # Gradient
    d_logits = softmax.copy()
    d_logits[np.arange(N), targets] -= 1.0
    d_logits /= N

    return loss, d_logits


# ── Adam optimizer ─────────────────────────────────────────────────────────────

class Adam:
    """
    Adam optimizer (Kingma & Ba, 2015).

    Maintains per-parameter first and second moment estimates with bias correction.
    """

    def __init__(self, lr: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0
        self._m    = []
        self._v    = []

    def step(self, params_and_grads: list):
        """
        Update parameters using Adam.

        params_and_grads: list of (param_array, grad_array)
        """
        if not self._m:
            self._m = [np.zeros_like(p) for p, g in params_and_grads]
            self._v = [np.zeros_like(p) for p, g in params_and_grads]

        self.t += 1
        b1t = 1 - self.beta1 ** self.t
        b2t = 1 - self.beta2 ** self.t

        for i, (p, g) in enumerate(params_and_grads):
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * g * g
            m_hat = self._m[i] / b1t
            v_hat = self._v[i] / b2t
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from conv_layer import Conv2d
    from pooling import MaxPool2d
    from flatten import Flatten

    print("=== Sequential CNN smoke test ===")
    np.random.seed(42)

    model = Sequential([
        Conv2d(1, 4, 3, padding=1),
        ReLU(),
        MaxPool2d(2),
        Conv2d(4, 8, 3, padding=1),
        ReLU(),
        MaxPool2d(2),
        Flatten(),
        Linear(8 * 7 * 7, 10),
    ])
    print(model)

    x      = np.random.randn(4, 1, 28, 28)
    logits = model.forward(x)
    print(f"Input:   {x.shape}")
    print(f"Logits:  {logits.shape}")

    targets = np.array([0, 3, 7, 9])
    loss, d_logits = cross_entropy_loss(logits, targets)
    print(f"Loss: {loss:.4f}")

    model.backward(d_logits)

    n_params = sum(p.size for p, g in model.parameters())
    print(f"Parameters: {n_params:,}")
    print("Sequential smoke test passed.")
