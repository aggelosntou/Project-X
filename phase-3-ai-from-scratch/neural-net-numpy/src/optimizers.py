"""
optimizers.py — Gradient descent optimizers operating on NumPy arrays.

Each optimizer takes a reference to the network's parameter list and
updates the parameter values (W, b) in-place after each backward pass.
"""

import numpy as np
from typing import Iterable, Tuple


class SGD:
    """
    Stochastic Gradient Descent.
    theta = theta - lr * grad
    """

    def __init__(self, network, lr: float = 0.01):
        self.network = network
        self.lr = lr

    def step(self):
        for (param, grad) in self.network.parameters():
            param -= self.lr * grad

    def zero_grad(self):
        for (_, grad) in self.network.parameters():
            grad[:] = 0


class SGDMomentum:
    """
    SGD with Momentum.
    v = momentum * v + grad
    theta = theta - lr * v

    Momentum "smooths" noisy gradients and accelerates in consistent directions.
    """

    def __init__(self, network, lr: float = 0.01, momentum: float = 0.9):
        self.network  = network
        self.lr       = lr
        self.momentum = momentum
        # Build velocity buffers matching parameter shapes
        self.velocities = [np.zeros_like(p) for p, _ in network.parameters()]

    def step(self):
        for i, (param, grad) in enumerate(self.network.parameters()):
            self.velocities[i] = self.momentum * self.velocities[i] + grad
            param -= self.lr * self.velocities[i]

    def zero_grad(self):
        for (_, grad) in self.network.parameters():
            grad[:] = 0


class Adam:
    """
    Adam: Adaptive Moment Estimation.
    Kingma & Ba, 2014.

    m = beta1 * m + (1-beta1) * grad
    v = beta2 * v + (1-beta2) * grad^2
    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)
    theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, network, lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.network = network
        self.lr      = lr
        self.beta1   = beta1
        self.beta2   = beta2
        self.eps     = eps
        self.t       = 0
        params = list(network.parameters())
        self.m = [np.zeros_like(p) for p, _ in params]
        self.v = [np.zeros_like(p) for p, _ in params]

    def step(self):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bc1 = 1 - b1 ** self.t
        bc2 = 1 - b2 ** self.t

        for i, (param, grad) in enumerate(self.network.parameters()):
            self.m[i] = b1 * self.m[i] + (1 - b1) * grad
            self.v[i] = b2 * self.v[i] + (1 - b2) * grad ** 2
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for (_, grad) in self.network.parameters():
            grad[:] = 0
