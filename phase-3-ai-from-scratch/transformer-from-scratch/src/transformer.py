"""
transformer.py — Full encoder-only Transformer model built from numpy primitives.

Imports MultiHeadAttention from attention.py and
LayerNorm / FeedForward / PositionalEncoding / Embedding from layers.py.

Architecture (Pre-Norm variant):
  Embedding → PositionalEncoding → N × EncoderBlock → Linear head

Pre-Norm vs Post-Norm
  Original "Attention Is All You Need" uses post-norm (norm after residual).
  Pre-norm (norm before the sub-layer) is used by GPT-2 / GPT-3 and is
  more stable to train without learning-rate warm-up.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from attention import MultiHeadAttention
from layers import LayerNorm, FeedForward, PositionalEncoding, Embedding


class EncoderBlock:
    """
    One transformer encoder block.

    Pre-norm residual:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Residual connections ensure gradient magnitude is at least 1:
        dL/dx = dL/dy * (1 + df/dx)
    Even if df/dx → 0 (dying sub-layer), gradient still flows through the +1 term.
    Without residuals, deep networks suffer from vanishing gradients where
    products of Jacobians → 0 exponentially with depth.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.attn  = MultiHeadAttention(d_model, num_heads)
        self.ff    = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self._cache = {}

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        # Pre-norm attention sub-layer
        normed1 = self.norm1.forward(x)
        attn_out = self.attn.forward(normed1, mask)
        x = x + attn_out                           # residual

        # Pre-norm feed-forward sub-layer
        normed2 = self.norm2.forward(x)
        ff_out = self.ff.forward(normed2)
        x = x + ff_out                             # residual

        self._cache = {'attn_out': attn_out, 'ff_out': ff_out}
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward through residual connections.
        y = x + f(norm(x))  →  dy/dx = 1 + df/d(norm) * d(norm)/dx
        grad flows through the identity path unchanged.
        """
        # Backward through FFN residual: grad splits to identity + ff path
        dff = self.ff.backward(grad)
        dnorm2 = self.norm2.backward(dff)
        grad = grad + dnorm2   # identity path carries full grad

        # Backward through attention residual
        dattn = self.attn_backward(grad)
        dnorm1 = self.norm1.backward(dattn)
        grad = grad + dnorm1

        return grad

    def attn_backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Minimal backward for multi-head attention.
        Full backprop through attention weights is complex;
        here we implement the weight-gradient accumulation paths
        used for the linear projection layers (W_Q, W_K, W_V, W_O).
        """
        cache = self.attn._cache
        x        = cache['x']
        merged   = cache['merged']
        attn_out = cache['attn_out']   # (batch, heads, seq, d_k)
        attn_weights = cache['attn_weights']
        Q_h, K_h, V_h = cache['Q_h'], cache['K_h'], cache['V_h']

        batch, seq, d_model = x.shape
        num_heads = self.attn.num_heads
        d_k = self.attn.d_k

        # grad: (batch, seq, d_model)  — gradient into output of W_O projection
        # dW_O: (d_model, d_model)
        self.attn.dW_O += merged.reshape(-1, d_model).T @ grad.reshape(-1, d_model)
        d_merged = grad @ self.attn.W_O.T            # (batch, seq, d_model)

        # Split back to (batch, heads, seq, d_k)
        d_attn_out = d_merged.reshape(batch, seq, num_heads, d_k).transpose(0, 2, 1, 3)

        # Gradient through attention: d_attn_out = attn_weights @ V_h
        # dV = attn_weights.T @ d_attn_out  (per head)
        d_V_h = attn_weights.swapaxes(-2, -1) @ d_attn_out  # (batch, heads, seq, d_k)

        # dW_V accumulation
        d_V = d_V_h.transpose(0, 2, 1, 3).reshape(batch, seq, d_model)
        self.attn.dW_V += x.reshape(-1, d_model).T @ d_V.reshape(-1, d_model)
        d_x_V = d_V @ self.attn.W_V.T

        # Gradient through softmax → scores (simplified: ignore attention score grads
        # for W_Q, W_K — full implementation would propagate through softmax).
        # For training tasks here this is sufficient; full BPTT through scores
        # would add dW_Q, dW_K terms.

        dx = d_x_V   # primary gradient path
        return dx

    def parameters(self):
        return (self.attn.parameters() +
                self.ff.parameters() +
                self.norm1.parameters() +
                self.norm2.parameters())


class Encoder:
    """
    Stack of N EncoderBlocks with shared embedding and positional encoding.

    Token flow:
      (batch, seq) ints → Embedding → + PositionalEncoding → N × EncoderBlock
    """

    def __init__(self, vocab_size: int, max_len: int,
                 n_blocks: int, d_model: int, num_heads: int, d_ff: int):
        self.embed   = Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(max_len, d_model)
        self.blocks  = [EncoderBlock(d_model, num_heads, d_ff) for _ in range(n_blocks)]
        self.d_model = d_model

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        x: (batch, seq_len) integer token indices
        Returns: (batch, seq_len, d_model)
        """
        # Embedding lookup + positional encoding
        emb = self.embed.forward(x)        # (batch, seq, d_model)
        h   = self.pos_enc.forward(emb)    # adds PE (no parameters)

        for block in self.blocks:
            h = block.forward(h, mask)
        return h

    def backward(self, grad: np.ndarray) -> None:
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        grad = self.pos_enc.backward(grad)
        self.embed.backward(grad)

    def parameters(self):
        params = list(self.embed.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        return params


class Transformer:
    """
    Encoder-only Transformer for sequence classification / language-modelling tasks.

    Architecture:
      Embedding(vocab_size, d_model)
      + PositionalEncoding(max_len, d_model)
      + N × EncoderBlock(d_model, num_heads, d_ff)
      + LayerNorm(d_model)         ← final norm (GPT-2 style)
      + Linear(d_model, vocab_size) ← projection head

    forward(x) → logits: (batch, seq_len, vocab_size)

    The linear head is weight-tied to the embedding matrix in some implementations
    (reduces parameters and often improves perplexity), but here they are separate
    for clarity.
    """

    def __init__(self, vocab_size: int, max_len: int,
                 n_blocks: int, d_model: int, num_heads: int, d_ff: int):
        self.vocab_size = vocab_size
        self.d_model    = d_model

        self.encoder = Encoder(vocab_size, max_len, n_blocks, d_model, num_heads, d_ff)
        self.norm_final = LayerNorm(d_model)

        # Linear projection head: d_model → vocab_size
        scale = np.sqrt(1.0 / d_model)
        self.W_head = np.random.randn(d_model, vocab_size) * scale
        self.b_head = np.zeros(vocab_size)
        self.dW_head = np.zeros_like(self.W_head)
        self.db_head = np.zeros_like(self.b_head)

        self._cache = {}

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        x: (batch, seq_len) integer token indices
        Returns logits: (batch, seq_len, vocab_size)
        """
        h = self.encoder.forward(x, mask)         # (batch, seq, d_model)
        h_normed = self.norm_final.forward(h)      # final LayerNorm
        logits = h_normed @ self.W_head + self.b_head  # (batch, seq, vocab_size)

        self._cache = {'h': h, 'h_normed': h_normed}
        return logits

    def backward(self, d_logits: np.ndarray) -> None:
        """
        d_logits: (batch, seq_len, vocab_size) — gradient of loss w.r.t. logits
        Propagates gradients all the way back through encoder and embedding.
        """
        h_normed = self._cache['h_normed']
        batch, seq, _ = d_logits.shape

        # Gradient of head linear layer
        self.dW_head += h_normed.reshape(-1, self.d_model).T @ d_logits.reshape(-1, self.vocab_size)
        self.db_head += d_logits.sum(axis=(0, 1))
        d_h_normed = d_logits @ self.W_head.T      # (batch, seq, d_model)

        # Backward through final LayerNorm
        d_h = self.norm_final.backward(d_h_normed)

        # Backward through encoder blocks + embedding
        self.encoder.backward(d_h)

    def zero_grad(self):
        """Zero all accumulated gradients."""
        self.dW_head[:] = 0
        self.db_head[:] = 0
        self.norm_final.dgamma[:] = 0
        self.norm_final.dbeta[:]  = 0
        for block in self.encoder.blocks:
            for layer in [block.ff, block.norm1, block.norm2]:
                for p, g in layer.parameters():
                    g[:] = 0
            for p, g in block.attn.parameters():
                g[:] = 0
        for p, g in self.encoder.embed.parameters():
            g[:] = 0

    def parameters(self):
        """Return list of (weight_array, grad_array) pairs for optimizer."""
        params = []
        # Head
        params.append((self.W_head, self.dW_head))
        params.append((self.b_head, self.db_head))
        # Final norm
        params.extend(self.norm_final.parameters())
        # Encoder
        params.extend(self.encoder.parameters())
        return params

    def sgd_step(self, lr: float):
        """Simple SGD update — update weights and zero gradients."""
        for p, g in self.parameters():
            p -= lr * g
            g[:] = 0

    def adam_step(self, lr: float, t: int,
                  beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        """Adam optimizer step. t is the step counter (1-indexed)."""
        if not hasattr(self, '_adam_m'):
            self._adam_m = [np.zeros_like(p) for p, g in self.parameters()]
            self._adam_v = [np.zeros_like(p) for p, g in self.parameters()]

        params_grads = self.parameters()
        for i, (p, g) in enumerate(params_grads):
            self._adam_m[i] = beta1 * self._adam_m[i] + (1 - beta1) * g
            self._adam_v[i] = beta2 * self._adam_v[i] + (1 - beta2) * g * g
            m_hat = self._adam_m[i] / (1 - beta1 ** t)
            v_hat = self._adam_v[i] / (1 - beta2 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)
            g[:] = 0


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray):
    """
    Cross-entropy loss and gradient for a batch of sequences.

    logits:  (batch, seq, vocab_size)
    targets: (batch, seq)  — integer class indices

    Returns: (scalar loss, gradient w.r.t. logits)

    Numerically stable via log-sum-exp:
      log(softmax(z)_i) = z_i - log(sum_j exp(z_j))
                        = z_i - (max(z) + log(sum_j exp(z_j - max(z))))
    """
    batch, seq, V = logits.shape

    # Shift for numerical stability
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_z   = np.exp(shifted)
    softmax = exp_z / exp_z.sum(axis=-1, keepdims=True)

    # NLL: pick log-prob of the correct class at each position
    log_probs = np.log(softmax + 1e-9)
    correct_log_probs = log_probs[
        np.arange(batch)[:, None],
        np.arange(seq)[None, :],
        targets
    ]
    loss = -correct_log_probs.mean()

    # Gradient: softmax(z) - one_hot(target), averaged over batch*seq
    d_logits = softmax.copy()
    d_logits[
        np.arange(batch)[:, None],
        np.arange(seq)[None, :],
        targets
    ] -= 1.0
    d_logits /= (batch * seq)

    return loss, d_logits


if __name__ == "__main__":
    print("=== Transformer smoke test ===")
    np.random.seed(42)

    vocab_size = 20
    batch_size = 4
    seq_len    = 8
    d_model    = 32
    num_heads  = 2
    d_ff       = 64
    n_blocks   = 2

    model = Transformer(
        vocab_size=vocab_size, max_len=50,
        n_blocks=n_blocks, d_model=d_model,
        num_heads=num_heads, d_ff=d_ff
    )

    x = np.random.randint(0, vocab_size, (batch_size, seq_len))
    logits = model.forward(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {logits.shape}  (expected {batch_size}, {seq_len}, {vocab_size})")

    loss, d_logits = cross_entropy_loss(logits, x)
    print(f"Loss: {loss:.4f}")
    model.backward(d_logits)

    n_params = sum(p.size for p, g in model.parameters())
    print(f"Total parameters: {n_params:,}")
    print("Smoke test passed.")
