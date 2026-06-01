"""
network.py — Sequential neural network container.

Chains layers forward and backward, threading gradients through the network.
"""

import numpy as np
from typing import List, Any


class Sequential:
    """
    A sequential stack of layers.

    Forward: x → layer_1 → layer_2 → ... → layer_n → output
    Backward: loss_grad → layer_n.backward → ... → layer_1.backward

    Each layer.backward(grad) returns the gradient w.r.t. the layer's INPUT,
    which becomes the gradient w.r.t. the previous layer's OUTPUT.
    This is the chain rule: dL/dx = dL/dy * dy/dx.
    """

    def __init__(self, layers: List):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad: np.ndarray) -> np.ndarray:
        """
        Propagate gradient backward through all layers.
        loss_grad: dL/d(output of last layer)
        Returns: dL/d(input of first layer)
        """
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update(self, lr: float):
        """Apply SGD update to all layers with learnable parameters."""
        for layer in self.layers:
            layer.update(lr)

    def parameters(self):
        """Yield (param, grad) tuples for all learnable parameters."""
        for layer in self.layers:
            for param_grad in layer.parameters():
                yield param_grad
