"""
attention.py — Scaled dot-product attention and multi-head attention.

The attention mechanism is the core innovation of the Transformer.
It allows every position in a sequence to directly attend to every other
position, regardless of distance — unlike RNNs where information must
flow through many sequential steps.

Key insight: attention computes a weighted sum of values, where weights
are determined by the similarity between queries and keys.
"""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along the given axis."""
    x_shifted = x - x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray = None
) -> tuple:
    """
    Scaled dot-product attention.

    Inputs (all with batch dimensions):
      Q: (..., seq_q, d_k)  — query matrix
      K: (..., seq_k, d_k)  — key matrix
      V: (..., seq_k, d_v)  — value matrix

    Output:
      attn_output: (..., seq_q, d_v)
      attn_weights: (..., seq_q, seq_k)  [for visualization/inspection]

    Math:
      Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Why divide by sqrt(d_k)?
      Q K^T computes dot products between all query-key pairs.
      With d_k-dimensional vectors (each component ~ N(0,1)):
        E[q · k] = 0
        Var[q · k] = d_k   (sum of d_k independent unit-variance terms)
        std[q · k] = sqrt(d_k)

      Without scaling, the dot products grow with sqrt(d_k), pushing
      softmax into the saturation region (near 0 or 1 gradients).
      Dividing by sqrt(d_k) normalizes variance back to 1.

    Why is attention permutation-equivariant?
      Each output position q attends to all keys k independently.
      If you permute the input sequence, the output is permuted the same way.
      That's why positional encoding is essential — without it, the model
      can't distinguish position 1 from position 10.
    """
    d_k = Q.shape[-1]
    scale = np.sqrt(d_k)

    # scores: (..., seq_q, seq_k)
    # Q K^T: for each query, dot product with each key
    scores = np.matmul(Q, K.swapaxes(-2, -1)) / scale

    # Mask: set masked positions to -inf so softmax assigns 0 probability.
    # In a decoder, we mask future positions (causal mask).
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    attn_weights = softmax(scores, axis=-1)

    # Weighted sum of values
    output = np.matmul(attn_weights, V)

    return output, attn_weights


class MultiHeadAttention:
    """
    Multi-head attention: run attention h times in parallel with different projections.

    For each head i:
      Q_i = Q W_Q^i,  K_i = K W_K^i,  V_i = V W_V^i
      head_i = Attention(Q_i, K_i, V_i)

    Concat all heads and project:
      MultiHead(Q,K,V) = concat(head_1, ..., head_h) W_O

    Why multiple heads?
      Each head can attend to different aspects of the input:
      - Head 1: might track syntactic dependencies
      - Head 2: might track coreference
      - Head 3: might track positional relationships
      etc.
      With a single head, all d_model dimensions compete for one attention pattern.
      With h heads, each head specializes in d_model/h dimensions.

    Parameters: d_model, num_heads (d_model must be divisible by num_heads)
    """

    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads  # dimension per head

        # Projection matrices: map d_model → d_model (split into heads inside forward)
        scale = 0.02
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

        # Gradient buffers
        self.dW_Q = np.zeros_like(self.W_Q)
        self.dW_K = np.zeros_like(self.W_K)
        self.dW_V = np.zeros_like(self.W_V)
        self.dW_O = np.zeros_like(self.W_O)

        # Cache for backward
        self._cache = {}

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Split the last dimension into (num_heads, d_k).
        (batch, seq, d_model) → (batch, num_heads, seq, d_k)
        """
        batch, seq, _ = x.shape
        x = x.reshape(batch, seq, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch, heads, seq, d_k)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Inverse of _split_heads.
        (batch, num_heads, seq, d_k) → (batch, seq, d_model)
        """
        batch, _, seq, _ = x.shape
        x = x.transpose(0, 2, 1, 3)  # (batch, seq, heads, d_k)
        return x.reshape(batch, seq, self.d_model)

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        x: (batch, seq, d_model)
        Self-attention: Q = K = V = x projected.
        For cross-attention, call with separate q, k, v inputs.
        """
        batch, seq, _ = x.shape

        # Project to Q, K, V spaces
        Q = x @ self.W_Q   # (batch, seq, d_model)
        K = x @ self.W_K
        V = x @ self.W_V

        # Split into heads
        Q_h = self._split_heads(Q)  # (batch, heads, seq, d_k)
        K_h = self._split_heads(K)
        V_h = self._split_heads(V)

        # Attention per head
        attn_out, attn_weights = scaled_dot_product_attention(Q_h, K_h, V_h, mask)
        # attn_out: (batch, heads, seq, d_k)

        # Merge heads and project
        merged = self._merge_heads(attn_out)    # (batch, seq, d_model)
        output = merged @ self.W_O              # (batch, seq, d_model)

        # Cache for backward
        self._cache = {
            'x': x, 'Q': Q, 'K': K, 'V': V,
            'Q_h': Q_h, 'K_h': K_h, 'V_h': V_h,
            'attn_out': attn_out, 'attn_weights': attn_weights,
            'merged': merged
        }

        return output

    def forward_cross(self, q: np.ndarray, kv: np.ndarray,
                      mask: np.ndarray = None) -> np.ndarray:
        """
        Cross-attention: queries come from decoder, keys/values from encoder.
        q:  (batch, tgt_seq, d_model)
        kv: (batch, src_seq, d_model)
        """
        Q = q  @ self.W_Q
        K = kv @ self.W_K
        V = kv @ self.W_V

        Q_h = self._split_heads(Q)
        K_h = self._split_heads(K)
        V_h = self._split_heads(V)

        attn_out, attn_weights = scaled_dot_product_attention(Q_h, K_h, V_h, mask)
        merged = self._merge_heads(attn_out)
        output = merged @ self.W_O

        self._cache = {
            'x': q, 'kv': kv, 'Q': Q, 'K': K, 'V': V,
            'Q_h': Q_h, 'K_h': K_h, 'V_h': V_h,
            'attn_out': attn_out, 'attn_weights': attn_weights,
            'merged': merged
        }

        return output

    def parameters(self):
        return [
            (self.W_Q, self.dW_Q), (self.W_K, self.dW_K),
            (self.W_V, self.dW_V), (self.W_O, self.dW_O)
        ]
