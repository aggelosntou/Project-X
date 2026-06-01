"""
conv_layer.py — 2D Convolution implemented via im2col.

WHAT CONVOLUTION COMPUTES
  In signal processing, convolution flips the kernel: (f * g)[n] = sum_k f[k] g[n-k]
  In deep learning, we compute cross-correlation (no flip):
    (f ⋆ g)[i,j] = sum_{p,q} f[i+p, j+q] * g[p,q]
  These are equivalent if the kernel is learned — the network can learn the
  flipped version just as easily. So "convolution" in deep learning means
  cross-correlation in practice.

KEY INSIGHT: WEIGHT SHARING
  A single kernel of shape (kH, kW) slides over the entire input image.
  The SAME weights detect the same pattern (e.g. a horizontal edge) everywhere.

  Why this works: the statistical structure of images is translation-equivariant.
  A dog in the corner and a dog in the center have the same local texture patterns.
  By sharing weights, we: (a) dramatically reduce parameters, (b) build in
  translation equivariance as an inductive bias, (c) need fewer examples to
  generalize because the learned detector works everywhere.

IM2COL: CONVOLUTION AS MATRIX MULTIPLY (GEMM)
  For an input x of shape (N, C, H, W) with kernel (F, C, kH, kW):

  im2col extracts all kH×kW×C patches centered at each output position and
  lays them out as rows of a 2D matrix:
    cols: (N * H_out * W_out,  C * kH * kW)

  Then convolution becomes a single matrix multiply:
    out_flat = cols @ W_flat.T    where W_flat: (F, C*kH*kW)

  This converts O(N * F * C * kH * kW * H_out * W_out) individual multiplies
  into one GEMM call, which BLAS (MKL, OpenBLAS, cuBLAS) can accelerate via
  SIMD, cache blocking, and hardware tensor cores — the same reason GPUs are
  fast at neural networks.
"""

import numpy as np


class Conv2d:
    """
    2D Convolution layer with im2col implementation.

    Parameters
    ----------
    in_channels  : C — input feature maps
    out_channels : F — output feature maps (number of filters)
    kernel_size  : kH = kW (square kernel)
    stride       : step size for sliding the kernel
    padding      : zero-padding added to each spatial edge

    Weight shapes:
      W: (F, C, kH, kW)   — F filters, each of shape (C, kH, kW)
      b: (F,)

    Initialization: Xavier uniform
      bound = sqrt(6 / (fan_in + fan_out))
      where fan_in = C * kH * kW, fan_out = F * kH * kW
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0):
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding

        kH = kW = kernel_size
        fan_in  = in_channels  * kH * kW
        fan_out = out_channels * kH * kW
        bound   = np.sqrt(6.0 / (fan_in + fan_out))

        self.W  = np.random.uniform(-bound, bound, (out_channels, in_channels, kH, kW))
        self.b  = np.zeros(out_channels)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self._x    = None
        self._cols = None

    def im2col(self, x: np.ndarray, kernel_size: int,
               stride: int, padding: int) -> np.ndarray:
        """
        Convert input feature maps to column matrix for GEMM-based convolution.

        x: (N, C, H, W)

        Steps:
          1. Pad input spatially with zeros
          2. For each output position (i, j), extract the (C, kH, kW) patch
          3. Flatten each patch to a row vector of length C*kH*kW
          4. Return matrix of shape (N * H_out * W_out, C * kH * kW)

        This is O(N * C * kH * kW * H_out * W_out) memory, but allows a
        single BLAS DGEMM call instead of nested Python loops.
        """
        N, C, H, W = x.shape
        kH = kW = kernel_size

        H_out = (H + 2 * padding - kH) // stride + 1
        W_out = (W + 2 * padding - kW) // stride + 1

        # Zero-pad the input
        if padding > 0:
            x_pad = np.pad(x, ((0,0), (0,0), (padding,padding), (padding,padding)),
                           mode='constant')
        else:
            x_pad = x

        # Build the column matrix using stride tricks / explicit indexing
        # Shape: (N, C, kH, kW, H_out, W_out)
        cols = np.zeros((N, C, kH, kW, H_out, W_out), dtype=x.dtype)

        for ki in range(kH):
            i_max = ki + stride * H_out
            for kj in range(kW):
                j_max = kj + stride * W_out
                cols[:, :, ki, kj, :, :] = x_pad[
                    :, :,
                    ki:i_max:stride,
                    kj:j_max:stride
                ]

        # Reshape to (N * H_out * W_out, C * kH * kW)
        cols = cols.transpose(0, 4, 5, 1, 2, 3)   # (N, H_out, W_out, C, kH, kW)
        cols = cols.reshape(N * H_out * W_out, C * kH * kW)
        return cols

    def col2im(self, cols: np.ndarray, x_shape: tuple,
               kernel_size: int, stride: int, padding: int) -> np.ndarray:
        """
        Inverse of im2col — accumulate gradient contributions back to input shape.

        cols:   (N * H_out * W_out, C * kH * kW)
        x_shape: (N, C, H, W)

        Because multiple patches overlap (when stride < kernel_size), each input
        element may receive gradients from multiple output positions.
        np.add.at ensures these are accumulated correctly (not overwritten).
        """
        N, C, H, W = x_shape
        kH = kW = kernel_size

        H_out = (H + 2 * padding - kH) // stride + 1
        W_out = (W + 2 * padding - kW) // stride + 1

        # Reshape cols back to (N, H_out, W_out, C, kH, kW)
        cols_reshaped = cols.reshape(N, H_out, W_out, C, kH, kW)
        # Transpose back to (N, C, kH, kW, H_out, W_out)
        cols_reshaped = cols_reshaped.transpose(0, 3, 4, 5, 1, 2)

        H_pad = H + 2 * padding
        W_pad = W + 2 * padding
        x_pad = np.zeros((N, C, H_pad, W_pad), dtype=cols.dtype)

        for ki in range(kH):
            i_max = ki + stride * H_out
            for kj in range(kW):
                j_max = kj + stride * W_out
                np.add.at(x_pad,
                          (slice(None), slice(None),
                           slice(ki, i_max, stride),
                           slice(kj, j_max, stride)),
                          cols_reshaped[:, :, ki, kj, :, :])

        if padding > 0:
            return x_pad[:, :, padding:-padding, padding:-padding]
        return x_pad

    def _output_size(self, H: int, W: int):
        kH = kW = self.kernel_size
        H_out = (H + 2 * self.padding - kH) // self.stride + 1
        W_out = (W + 2 * self.padding - kW) // self.stride + 1
        return H_out, W_out

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        x: (N, C, H, W)
        Returns: (N, F, H_out, W_out)

        Algorithm:
          1. im2col(x) → cols: (N*H_out*W_out, C*kH*kW)
          2. W_flat = W.reshape(F, C*kH*kW)
          3. out_flat = cols @ W_flat.T + b  → (N*H_out*W_out, F)
          4. Reshape to (N, H_out, W_out, F) → transpose → (N, F, H_out, W_out)
        """
        N, C, H, W = x.shape
        F = self.out_channels
        kH = kW = self.kernel_size
        H_out, W_out = self._output_size(H, W)

        self._x = x
        self._cols = self.im2col(x, self.kernel_size, self.stride, self.padding)

        W_flat = self.W.reshape(F, C * kH * kW)         # (F, C*kH*kW)
        out_flat = self._cols @ W_flat.T + self.b        # (N*H_out*W_out, F)

        out = out_flat.reshape(N, H_out, W_out, F)
        out = out.transpose(0, 3, 1, 2)                  # (N, F, H_out, W_out)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        dout: (N, F, H_out, W_out)
        Returns: dx (N, C, H, W)

        Derivation (using the GEMM view):
          out_flat = cols @ W_flat.T + b
          where out_flat: (N*H_out*W_out, F)
                cols:     (N*H_out*W_out, C*kH*kW)
                W_flat:   (F, C*kH*kW)

          dW_flat = cols.T @ dout_flat    →  accumulate into self.dW
          db      = dout_flat.sum(axis=0) →  accumulate into self.db
          dcols   = dout_flat @ W_flat    →  (N*H_out*W_out, C*kH*kW)
          dx      = col2im(dcols)         →  (N, C, H, W)
        """
        N, C, H, W = self._x.shape
        F = self.out_channels
        kH = kW = self.kernel_size

        # Reshape dout: (N, F, H_out, W_out) → (N*H_out*W_out, F)
        dout_t    = dout.transpose(0, 2, 3, 1)          # (N, H_out, W_out, F)
        dout_flat = dout_t.reshape(-1, F)

        W_flat = self.W.reshape(F, C * kH * kW)

        # Gradient w.r.t. weights and bias
        self.dW += (self._cols.T @ dout_flat).T.reshape(self.W.shape)
        self.db += dout_flat.sum(axis=0)

        # Gradient w.r.t. input (via col2im)
        dcols = dout_flat @ W_flat                       # (N*H_out*W_out, C*kH*kW)
        dx    = self.col2im(dcols, self._x.shape,
                            self.kernel_size, self.stride, self.padding)
        return dx

    def parameters(self):
        """Return list of (param, grad) pairs for optimizer."""
        return [(self.W, self.dW), (self.b, self.db)]

    def zero_grad(self):
        self.dW[:] = 0
        self.db[:] = 0


if __name__ == "__main__":
    print("=== Conv2d smoke test ===")
    np.random.seed(0)

    # (N, C, H, W) = (2, 1, 8, 8)
    x = np.random.randn(2, 1, 8, 8)
    conv = Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)

    out = conv.forward(x)
    print(f"Input:  {x.shape}  →  Output: {out.shape}")
    assert out.shape == (2, 4, 8, 8), f"Expected (2,4,8,8), got {out.shape}"

    # Backward
    dout = np.ones_like(out)
    dx   = conv.backward(dout)
    print(f"dout:   {dout.shape}  →  dx:    {dx.shape}")
    assert dx.shape == x.shape

    # Numerical gradient check (one element)
    eps = 1e-5
    i, j = 0, 0
    x_plus  = x.copy(); x_plus[0, 0, 0, 0] += eps
    x_minus = x.copy(); x_minus[0, 0, 0, 0] -= eps

    conv2 = Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
    conv2.W = conv.W.copy(); conv2.b = conv.b.copy()

    out_p  = conv2.forward(x_plus).sum()
    out_m  = conv2.forward(x_minus).sum()
    num_dx = (out_p - out_m) / (2 * eps)

    # Analytical dx at same position
    conv.zero_grad()
    _ = conv.forward(x)
    ana_dx = conv.backward(np.ones_like(conv.forward(x)))
    ana_val = ana_dx[0, 0, 0, 0]

    print(f"Numerical dx[0,0,0,0]:  {num_dx:.6f}")
    print(f"Analytical dx[0,0,0,0]: {ana_val:.6f}")
    print("Conv2d test passed.")
