"""
losses.py — Loss functions with analytical gradient implementations.

Each loss class implements:
  - forward(pred, target): compute scalar loss
  - backward(): return dL/d(pred), the gradient w.r.t. predictions

Why implement both forward and backward?
The backward pass is where the learning signal comes from. The gradient
tells each parameter: "increase me and the loss increases, so decrease me."
"""

import numpy as np


class MSELoss:
    """
    Mean Squared Error: L = (1/N) * sum((pred - target)^2)

    This is the L2 loss. It penalizes large errors quadratically —
    an error twice as large gets penalized 4× more.

    Gradient:
        dL/d(pred) = (2/N) * (pred - target)

    Derivation:
        L = (1/N) sum_i (p_i - t_i)^2
        dL/dp_j = (1/N) * 2 * (p_j - t_j)
    """

    def __init__(self):
        self._pred = None
        self._target = None

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self._pred   = pred
        self._target = target
        return float(np.mean((pred - target) ** 2))

    def backward(self) -> np.ndarray:
        n = self._pred.shape[0]
        return 2 * (self._pred - self._target) / n


class CrossEntropyLoss:
    """
    Combined log-softmax + negative log-likelihood.

    This fuses softmax and cross-entropy into a single numerically-stable operation.

    For logits z (unnormalized) and one-hot targets t:
        L = -sum_i t_i * log(softmax(z)_i)
          = -sum_i t_i * (z_i - log(sum_j exp(z_j)))
          = log(sum_j exp(z_j)) - sum_i t_i * z_i

    Gradient (the elegant result):
        dL/dz_i = softmax(z)_i - t_i

    This is the most common output layer for classification. The gradient is
    simply the difference between predicted probability and target.

    Why fuse softmax and CE?
    If we compute them separately:
      1. softmax(1000) overflows
      2. log(softmax(x)) = log(exp(x-max)/sum) = (x-max) - log(sum exp(x-max))
         is computed more stably as a single operation (log-sum-exp trick).

    Parameters
    ----------
    labels: integer class indices (not one-hot) OR one-hot float array
    """

    def __init__(self):
        self._softmax = None
        self._target  = None

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        logits: (batch, n_classes), raw unnormalized scores
        labels: (batch,) integer class indices OR (batch, n_classes) one-hot
        """
        # Numerically stable softmax
        z = logits - logits.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        self._softmax = exp_z / exp_z.sum(axis=1, keepdims=True)

        n = logits.shape[0]
        n_cls = logits.shape[1]

        # Handle both integer labels and one-hot
        if labels.ndim == 1:
            # Convert to one-hot
            target = np.zeros((n, n_cls))
            target[np.arange(n), labels.astype(int)] = 1.0
        else:
            target = labels

        self._target = target

        # L = -mean(sum_j target_j * log(softmax_j))
        eps = 1e-12
        return float(-np.mean(np.sum(target * np.log(self._softmax + eps), axis=1)))

    def backward(self) -> np.ndarray:
        """
        Gradient: (softmax - target) / batch_size
        """
        n = self._softmax.shape[0]
        return (self._softmax - self._target) / n
