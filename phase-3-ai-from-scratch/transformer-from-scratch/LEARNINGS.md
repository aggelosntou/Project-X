# Transformer From Scratch — Key Learnings

## 1. Why Scaled Dot-Product Attention (the sqrt(d_k) factor)

Attention computes Q K^T, a matrix of dot products between every query-key pair.

If Q and K have components drawn independently from N(0,1), then each dot product
q · k is a sum of d_k independent unit-variance terms:

    E[q · k] = 0
    Var[q · k] = d_k          (sum of d_k variables each with variance 1)
    std[q · k] = sqrt(d_k)

Without scaling, as d_k grows the dot products grow proportionally to sqrt(d_k).
Large values push the softmax into the saturation region: when one score is much
larger than the others, softmax ≈ one-hot and its derivative ≈ 0.

Concretely, softmax(x)_i = exp(x_i) / sum_j exp(x_j), so:

    d(softmax)/dx_i = softmax_i * (1 - softmax_i)

When softmax_i → 1 or softmax_i → 0, this derivative → 0. Gradients vanish.
The network cannot learn which positions to attend to.

Dividing by sqrt(d_k) restores variance to ~1, keeping softmax in the linear
regime where gradients are healthy.

## 2. Why Multi-Head Attention

With a single attention head, all d_model dimensions must share one attention
pattern. The head can only learn one way to weigh positions at a time.

Multi-head attention projects Q, K, V into h subspaces of dimension d_k = d_model/h
and runs attention independently in each:

    head_i = Attention(Q W_Q^i, K W_K^i, V W_V^i)
    output = concat(head_1, ..., head_h) W_O

Different heads can specialize:
- Head 1 might track syntactic relationships (verb → subject agreement)
- Head 2 might track coreference ("the president ... he")
- Head 3 might track relative positions (adjacent tokens)
- Head 4 might track semantic similarity (synonyms)

This is only possible if heads have separate projection matrices. With one head,
you get one averaged attention pattern blending all these needs.

Crucially, the total parameter count and FLOPs are the same as one big head —
the computation is just re-distributed into parallel independent subspaces.

## 3. What Residual Connections Prevent (mathematically)

A residual connection defines: y = x + f(x)

The gradient of the loss L with respect to input x is:

    dL/dx = dL/dy * dy/dx = dL/dy * (1 + df/dx)

This has two terms:
- `1` — the identity path. Even if df/dx = 0 (e.g. dying ReLU, over-normalized layer),
         the gradient dL/dy passes through unchanged.
- `df/dx` — the learned transformation path.

In a deep network with N residual blocks, the gradient flows as a product:

    dL/dx_0 = dL/dx_N * product_{i=0}^{N-1} (1 + df_i/dx_i)

Even if each df_i/dx_i ≈ 0, the product ≥ 1, so deep networks can train.

Without residuals, the gradient is the product of N Jacobians. If each has
spectral norm < 1, the product → 0 exponentially (vanishing gradient).
If each has norm > 1, the product → ∞ exponentially (exploding gradient).
Residuals break this product structure by adding 1 at every layer.

## 4. Why Positional Encoding is Needed

Self-attention is permutation-equivariant: if you shuffle the input sequence,
the output is shuffled in exactly the same way. Formally:

    Attention(Px, Px, Px) = P * Attention(x, x, x)

where P is any permutation matrix.

This means the model has no way to distinguish "the cat sat" from "sat the cat"
without additional position information. The model sees only a set of tokens,
not an ordered sequence.

Positional encoding adds a unique "fingerprint" to each position:

    PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))

Why sinusoidal (not learned)?
1. Generalizes to sequences longer than seen during training (no lookup table overflow)
2. The relative offset between positions is expressible as a linear function:
   sin(pos + k) = sin(pos)cos(k) + cos(pos)sin(k)  — a linear function of sin(pos), cos(pos)
   So the model can learn "attend to the token 3 positions back" by learning the
   appropriate linear combination of the positional encoding dimensions.

Each dimension oscillates at a different period (from 2π to 2π × 10000),
like bits in a binary number — creating a unique pattern for every position.

## 5. Layer Normalization vs Batch Normalization

BatchNorm normalizes over the batch dimension (mean/var computed across the batch).
This requires large batches to estimate statistics accurately and doesn't work for
batch size 1. For variable-length sequences, batch statistics are ill-defined.

LayerNorm normalizes over the feature dimension (mean/var computed per token).
This works for any batch size, any sequence length, and is position-independent —
critical for the (batch, seq, d_model) layout of Transformers.

## 6. Pre-Norm vs Post-Norm

Post-Norm (original paper): x = LayerNorm(x + f(x))
Pre-Norm (GPT-2/3):         x = x + f(LayerNorm(x))

Pre-Norm is more stable because the residual path is always un-normalized.
Post-Norm can suffer from gradient instability in the early stages of training
because the LayerNorm wraps the residual sum, suppressing the identity gradient path.
GPT-2, GPT-3, and most modern Transformers use Pre-Norm.
