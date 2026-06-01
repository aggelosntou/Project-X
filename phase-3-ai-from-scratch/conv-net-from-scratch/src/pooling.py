"""
pooling.py — Max pooling and Average pooling layers.

WHAT MAX POOLING ACHIEVES
  1. Spatial invariance: if the strongest activation shifts by 1 pixel within
     a pool window, the output is unchanged — a rough form of translation invariance.
     (Translation equivariance from convolution + translation invariance from pooling
      ≈ the basis of why CNNs are robust to small spatial shifts.)

  2. Downsampling: reduces spatial resolution by a factor of stride, decreasing
     computation in subsequent layers and increasing the effective receptive field.

  3. Parameter-free: pooling has no learnable weights — it's a deterministic
     aggregation. This keeps the model compact.

BACKWARD THROUGH MAX POOLING
  Only the maximum element in each pool window receives gradient.
  All other elements receive zero gradient (they didn't affect the output).
  This is implemented by storing the argmax indices during forward and using
  them as a routing mask during backward.
"""

import numpy as np


class MaxPool2d:
    """
    Max pooling: takes the maximum value in each (kH × kW) window.

    Parameters
    ----------
    kernel_size : pool window size (square)
    stride      : step size (defaults to kernel_size → non-overlapping windows)

    Input:  (N, C, H, W)
    Output: (N, C, H_out, W_out)
      where H_out = (H - kH) // stride + 1
    """

    def __init__(self, kernel_size: int, stride: int = None):
        self.kernel_size = kernel_size
        self.stride      = stride if stride is not None else kernel_size

        self._x_shape  = None
        self._max_mask = None   # boolean mask: True at the max position

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (N, C, H, W)
        Returns: (N, C, H_out, W_out)
        """
        N, C, H, W = x.shape
        kH = kW = self.kernel_size
        s  = self.stride

        H_out = (H - kH) // s + 1
        W_out = (W - kW) // s + 1

        self._x_shape = x.shape

        # Extract patches: (N, C, H_out, W_out, kH, kW)
        # Using explicit loop over kernel positions for clarity
        patches = np.lib.stride_tricks.as_strided(
            x,
            shape   = (N, C, H_out, W_out, kH, kW),
            strides = (x.strides[0], x.strides[1],
                       x.strides[2] * s, x.strides[3] * s,
                       x.strides[2], x.strides[3])
        )
        # patches: (N, C, H_out, W_out, kH, kW)
        patches_flat = patches.reshape(N, C, H_out, W_out, kH * kW)

        # Argmax within each pool window
        argmax_flat = patches_flat.argmax(axis=-1)   # (N, C, H_out, W_out)
        out         = patches_flat.max(axis=-1)      # (N, C, H_out, W_out)

        # Build mask for backward: mark which element in each patch was the max
        self._max_mask  = (patches_flat == out[..., np.newaxis])  # (N,C,H_out,W_out,kH*kW)
        # In case of ties, only propagate to the first max
        first_max = np.zeros_like(self._max_mask)
        idx = np.argmax(self._max_mask, axis=-1, keepdims=True)
        np.put_along_axis(first_max, idx, True, axis=-1)
        self._max_mask = first_max

        self._H_out = H_out
        self._W_out = W_out

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        dout: (N, C, H_out, W_out)
        Returns: dx (N, C, H, W)

        Gradient routing:
          For each pool window, only the max element gets the gradient dout[i,j].
          All other elements get 0.

        We reconstruct dx by broadcasting dout[..., newaxis] through the mask
        and accumulating back into the input spatial positions.
        """
        N, C, H, W = self._x_shape
        kH = kW = self.kernel_size
        s  = self.stride
        H_out, W_out = self._H_out, self._W_out

        # Expand dout and route through mask
        # dout:      (N, C, H_out, W_out)      → (N, C, H_out, W_out, kH*kW)
        # max_mask:  (N, C, H_out, W_out, kH*kW)
        d_patches_flat = self._max_mask * dout[..., np.newaxis]
        d_patches = d_patches_flat.reshape(N, C, H_out, W_out, kH, kW)

        # Accumulate gradients back to original spatial positions
        dx = np.zeros(self._x_shape, dtype=dout.dtype)
        for ki in range(kH):
            for kj in range(kW):
                i_end = ki + s * H_out
                j_end = kj + s * W_out
                dx[:, :, ki:i_end:s, kj:j_end:s] += d_patches[:, :, :, :, ki, kj]

        return dx


class AvgPool2d:
    """
    Average pooling: takes the mean value in each (kH × kW) window.

    Backward: gradient distributes uniformly to all positions in the window.
    """

    def __init__(self, kernel_size: int, stride: int = None):
        self.kernel_size = kernel_size
        self.stride      = stride if stride is not None else kernel_size
        self._x_shape    = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        kH = kW = self.kernel_size
        s  = self.stride

        H_out = (H - kH) // s + 1
        W_out = (W - kW) // s + 1
        self._x_shape = x.shape
        self._H_out, self._W_out = H_out, W_out

        patches = np.lib.stride_tricks.as_strided(
            x,
            shape   = (N, C, H_out, W_out, kH, kW),
            strides = (x.strides[0], x.strides[1],
                       x.strides[2] * s, x.strides[3] * s,
                       x.strides[2], x.strides[3])
        )
        return patches.mean(axis=(-2, -1))

    def backward(self, dout: np.ndarray) -> np.ndarray:
        N, C, H, W = self._x_shape
        kH = kW = self.kernel_size
        s  = self.stride
        H_out, W_out = self._H_out, self._W_out

        dx = np.zeros(self._x_shape, dtype=dout.dtype)
        d_per_element = dout / (kH * kW)   # uniform distribution

        for ki in range(kH):
            for kj in range(kW):
                i_end = ki + s * H_out
                j_end = kj + s * W_out
                dx[:, :, ki:i_end:s, kj:j_end:s] += d_per_element

        return dx


if __name__ == "__main__":
    print("=== MaxPool2d smoke test ===")
    np.random.seed(42)
    x = np.random.randn(2, 3, 8, 8)

    pool = MaxPool2d(kernel_size=2, stride=2)
    out  = pool.forward(x)
    print(f"Input:  {x.shape}  →  Output: {out.shape}")
    assert out.shape == (2, 3, 4, 4)

    dout = np.ones_like(out)
    dx   = pool.backward(dout)
    print(f"dx shape: {dx.shape}")
    assert dx.shape == x.shape

    # Check gradient sum: each input should receive grad in exactly one pool window
    # Sum of dx should equal sum of dout
    print(f"Sum(dout)={dout.sum():.1f}, Sum(dx)={dx.sum():.1f}")
    assert abs(dout.sum() - dx.sum()) < 1e-9, "Gradient sum mismatch!"

    # Numerical gradient check (single element)
    eps = 1e-5
    x_copy = x.copy()
    x_plus = x.copy(); x_plus[0, 0, 0, 0] += eps
    x_minus = x.copy(); x_minus[0, 0, 0, 0] -= eps

    pool2 = MaxPool2d(kernel_size=2, stride=2)
    out_p = pool2.forward(x_plus).sum()
    pool3 = MaxPool2d(kernel_size=2, stride=2)
    out_m = pool3.forward(x_minus).sum()
    num_grad = (out_p - out_m) / (2 * eps)

    # Analytical gradient at [0,0,0,0]
    pool4 = MaxPool2d(kernel_size=2, stride=2)
    _ = pool4.forward(x_copy)
    ana_dx = pool4.backward(np.ones_like(out))
    ana_val = ana_dx[0, 0, 0, 0]

    print(f"Numerical  grad[0,0,0,0]: {num_grad:.6f}")
    print(f"Analytical grad[0,0,0,0]: {ana_val:.6f}")
    print("MaxPool2d test passed.")
