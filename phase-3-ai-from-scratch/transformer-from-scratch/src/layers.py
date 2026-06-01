"""
layers.py — Transformer building blocks: LayerNorm, FeedForward,
            PositionalEncoding, Embedding.
"""

import numpy as np


class LayerNorm:
    """
    Layer Normalization (Ba et al., 2016).

    Normalizes the last dimension (feature dimension) of the input,
    then applies learnable scale (gamma) and shift (beta).

    Formula:
        mean = mean(x, axis=-1)
        var  = var(x, axis=-1)
        x_hat = (x - mean) / sqrt(var + eps)
        output = gamma * x_hat + beta

    Why LayerNorm (not BatchNorm) in transformers?
      BatchNorm normalizes over the BATCH dimension — requires large batches
      and doesn't work well for variable-length sequences.
      LayerNorm normalizes over the FEATURE dimension — works for any batch
      size (even 1) and is independent of sequence length. Much more natural
      for the (batch, seq, d_model) layout of transformers.

    Backward:
      This is the most involved gradient in the transformer.
      Let x_hat = (x - mean) / std, y = gamma * x_hat + beta.
      dL/dx involves three terms (via mean, std, and direct path).
      See implementation below.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps     = eps
        self.gamma   = np.ones(d_model)    # scale — starts at 1
        self.beta    = np.zeros(d_model)   # shift — starts at 0

        self.dgamma  = np.zeros(d_model)
        self.dbeta   = np.zeros(d_model)
        self._cache  = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var  = x.var(axis=-1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        out = self.gamma * x_hat + self.beta

        self._cache = {'x': x, 'mean': mean, 'var': var, 'x_hat': x_hat}
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        Backward through LayerNorm.

        Let n = d_model (number of features being normalized).
        d = x - mean,   std = sqrt(var + eps)

        dL/dgamma = sum(grad_out * x_hat, axis=reduction_axes)
        dL/dbeta  = sum(grad_out, axis=reduction_axes)

        dL/dx (the hard part, derived by unrolling the mean and var):
          dL/dx_hat = grad_out * gamma
          dL/dvar   = sum(dL/dx_hat * d * -0.5 * std^{-3})
          dL/dmean  = sum(dL/dx_hat * -1/std) + dL/dvar * mean(-2d)
          dL/dx     = dL/dx_hat / std + dL/dvar * 2d/n + dL/dmean / n
        """
        x, mean, var, x_hat = (self._cache[k] for k in ('x', 'mean', 'var', 'x_hat'))
        n = self.d_model
        std = np.sqrt(var + self.eps)
        d   = x - mean

        # Gradient of learnable params (sum over all dimensions except d_model)
        self.dgamma += (grad_out * x_hat).sum(axis=tuple(range(grad_out.ndim - 1)))
        self.dbeta  += grad_out.sum(axis=tuple(range(grad_out.ndim - 1)))

        # Gradient w.r.t. x
        dxhat = grad_out * self.gamma
        dvar  = (dxhat * d * (-0.5) * (var + self.eps) ** (-1.5)).sum(axis=-1, keepdims=True)
        dmean = (dxhat * (-1 / std)).sum(axis=-1, keepdims=True) + dvar * d.mean(axis=-1, keepdims=True) * (-2)
        dx    = dxhat / std + dvar * 2 * d / n + dmean / n

        return dx

    def update(self, lr: float):
        self.gamma -= lr * self.dgamma
        self.beta  -= lr * self.dbeta
        self.dgamma[:] = 0
        self.dbeta[:]  = 0

    def parameters(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]


class FeedForward:
    """
    Position-wise feed-forward network.

    Applied independently to each position:
        FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

    In standard transformers: d_ff = 4 * d_model.
    This "bottleneck expand-compress" gives the network capacity to
    process information non-linearly without the overhead of full O(d_ff^2).
    """

    def __init__(self, d_model: int, d_ff: int):
        scale = np.sqrt(2.0 / (d_model + d_ff))
        self.W1 = np.random.randn(d_model, d_ff) * scale
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale
        self.b2 = np.zeros(d_model)

        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)

        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        out = a1 @ self.W2 + self.b2

        self._cache = {'x': x, 'z1': z1, 'a1': a1}
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x, z1, a1 = self._cache['x'], self._cache['z1'], self._cache['a1']

        # Backward through W2
        self.dW2 += a1.reshape(-1, a1.shape[-1]).T @ grad_out.reshape(-1, grad_out.shape[-1])
        self.db2 += grad_out.sum(axis=tuple(range(grad_out.ndim - 1)))
        da1 = grad_out @ self.W2.T

        # Backward through ReLU
        dz1 = da1 * (z1 > 0)

        # Backward through W1
        self.dW1 += x.reshape(-1, x.shape[-1]).T @ dz1.reshape(-1, dz1.shape[-1])
        self.db1 += dz1.sum(axis=tuple(range(dz1.ndim - 1)))
        dx = dz1 @ self.W1.T

        return dx

    def update(self, lr: float):
        self.W1 -= lr * self.dW1; self.dW1[:] = 0
        self.b1 -= lr * self.db1; self.db1[:] = 0
        self.W2 -= lr * self.dW2; self.dW2[:] = 0
        self.b2 -= lr * self.db2; self.db2[:] = 0

    def parameters(self):
        return [(self.W1, self.dW1), (self.b1, self.db1),
                (self.W2, self.dW2), (self.b2, self.db2)]


class PositionalEncoding:
    """
    Sinusoidal positional encoding.

    PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))

    Why sin/cos at different frequencies?
      Each dimension of the encoding oscillates at a different period:
      - Dimension 0: period 2π      (changes every ~6 positions)
      - Dimension d_model-2: period 2π * 10000  (barely changes over 10k positions)

      This creates a unique "fingerprint" for each position, similar to how
      binary numbers encode integers (but continuous and smooth).

    Why sinusoidal (not learned)?
      - Can generalize to sequences longer than seen during training.
      - The relative position (pos + k) can be expressed as a linear
        function of position (pos), because sin(a+b) = sin(a)cos(b) + cos(a)sin(b).
        So the model can learn to attend to "relative" positions.

    No parameters to learn — this is a fixed lookup table added to embeddings.
    """

    def __init__(self, max_len: int, d_model: int):
        self.d_model = d_model

        # Compute PE table
        pe = np.zeros((max_len, d_model))
        pos = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)
        i   = np.arange(0, d_model, 2)           # even indices: 0, 2, 4, ...

        # Division term: 10000^(2i/d_model)
        div_term = np.power(10000.0, 2 * i / d_model)

        pe[:, 0::2] = np.sin(pos / div_term)   # even dimensions
        pe[:, 1::2] = np.cos(pos / div_term[:d_model//2])   # odd dimensions

        self.pe = pe  # (max_len, d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to token embeddings.
        x: (batch, seq, d_model)
        """
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :]  # broadcast over batch

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Gradient passes through unchanged (PE has no parameters)."""
        return grad_out


class Embedding:
    """
    Token embedding table: maps integer token IDs to dense vectors.

    Parameters: (vocab_size, d_model) matrix.
    Row i is the embedding vector for token i.

    Backward: the gradient w.r.t. the embedding table is sparse —
    only the embeddings for tokens that appeared in the batch get updated.
    """

    def __init__(self, vocab_size: int, d_model: int):
        # Small random init (not Xavier — embedding lookup is not a linear layer)
        self.W = np.random.randn(vocab_size, d_model) * 0.01
        self.dW = np.zeros_like(self.W)
        self._tokens = None

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """
        tokens: (batch, seq) integer array
        Returns: (batch, seq, d_model)
        """
        self._tokens = tokens
        return self.W[tokens]

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        Accumulate gradients for the embedding rows that were used.
        This is a sparse update: we add grad_out[b, t] to dW[token[b, t]].
        """
        np.add.at(self.dW, self._tokens, grad_out)
        return None  # no gradient w.r.t. token indices (they're discrete)

    def update(self, lr: float):
        self.W -= lr * self.dW
        self.dW[:] = 0

    def parameters(self):
        return [(self.W, self.dW)]
