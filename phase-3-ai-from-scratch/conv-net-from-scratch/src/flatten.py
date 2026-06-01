"""
flatten.py — Flatten layer connecting convolutional and fully-connected layers.

Bridges the (N, C, H, W) world of convolution with the (N, features) world
of linear layers. Stores the input shape to correctly invert during backward.
"""

import numpy as np


class Flatten:
    """
    Flatten all dimensions except the batch dimension.

    forward:  (N, C, H, W)  →  (N, C*H*W)
    backward: (N, C*H*W)    →  (N, C, H, W)

    No learnable parameters.
    """

    def __init__(self):
        self._shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(self._shape)

    def parameters(self):
        return []   # no parameters


if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.randn(4, 8, 7, 7)
    flat = Flatten()
    out  = flat.forward(x)
    print(f"Input:  {x.shape}  →  Output: {out.shape}")
    assert out.shape == (4, 8 * 7 * 7)

    dout = np.random.randn(*out.shape)
    dx   = flat.backward(dout)
    print(f"dout:   {dout.shape}  →  dx:    {dx.shape}")
    assert dx.shape == x.shape
    print("Flatten test passed.")
