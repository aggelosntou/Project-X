"""
flops_counter.py — FLOPs (floating-point operations) counter for ML layers.

A FLOPs count answers: "How much arithmetic does one forward pass do?"
This is useful for:
  - Comparing model sizes beyond just parameter count
  - Estimating achievable throughput on a given accelerator
  - Identifying compute vs memory bottlenecks (roofline model)

Convention: multiply-accumulate = 2 FLOPs (one multiply + one add).
This is the standard convention used in most ML papers.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Layer-wise FLOPs formulas
# ---------------------------------------------------------------------------

def flops_linear(batch: int, in_features: int, out_features: int) -> int:
    """
    FLOPs for a fully-connected (linear) layer: y = x @ W.T + b

    Each output element requires in_features multiplications and
    in_features additions → 2 * in_features FLOPs.
    There are batch * out_features output elements.

    Total: 2 * batch * in_features * out_features

    The bias addition adds out_features FLOPs per sample, but this is
    negligible (in_features >> 1) and is usually omitted.

    Parameters
    ----------
    batch       : int — batch size
    in_features : int — input dimension
    out_features: int — output dimension

    Returns
    -------
    int — total FLOPs
    """
    return 2 * batch * in_features * out_features


def flops_conv(batch: int, in_channels: int, out_channels: int,
               kernel_size: int, H_out: int, W_out: int) -> int:
    """
    FLOPs for a 2D convolution layer (square kernel).

    Each output pixel (at position h, w for channel out_c) is the sum over
    in_channels × kernel_size × kernel_size multiply-accumulate operations.

    Total: 2 * batch * out_channels * H_out * W_out * in_channels * k * k

    Parameters
    ----------
    batch       : int
    in_channels : int
    out_channels: int
    kernel_size : int  (k; assumes square k×k kernel)
    H_out       : int  (output height after conv)
    W_out       : int  (output width after conv)

    Returns
    -------
    int — total FLOPs
    """
    k = kernel_size
    return 2 * batch * out_channels * H_out * W_out * in_channels * k * k


def flops_attention(batch: int, seq_len: int,
                    d_model: int, num_heads: int) -> int:
    """
    FLOPs for one transformer self-attention block.

    Breakdown:
      1. Q, K, V projections: 3 linear layers d_model → d_model
         = 3 * 2 * batch * seq_len * d_model * d_model

      2. Scaled dot-product: Q @ K^T  (batch, heads, seq, d_head)
         = 2 * batch * heads * seq^2 * d_head

      3. Attention-weighted sum: softmax(QK^T) @ V
         = 2 * batch * heads * seq^2 * d_head

      4. Output projection: linear d_model → d_model
         = 2 * batch * seq_len * d_model * d_model

    Note: softmax FLOPs (exp, div) are omitted — they are O(seq^2) but
    at much lower constant than GEMM and are typically memory-bound.

    Parameters
    ----------
    batch     : int
    seq_len   : int
    d_model   : int  (total embedding dimension)
    num_heads : int

    Returns
    -------
    int — total FLOPs
    """
    d_head = d_model // num_heads

    # 1. Q, K, V projections
    qkv_proj = 3 * 2 * batch * seq_len * d_model * d_model

    # 2. QK^T
    qkt = 2 * batch * num_heads * seq_len * seq_len * d_head

    # 3. attention × V
    av  = 2 * batch * num_heads * seq_len * seq_len * d_head

    # 4. Output projection
    out_proj = 2 * batch * seq_len * d_model * d_model

    return qkv_proj + qkt + av + out_proj


# ---------------------------------------------------------------------------
# Model-level helpers
# ---------------------------------------------------------------------------

def count_model_flops(layer_sizes: list, batch_size: int = 32) -> int:
    """
    Count total FLOPs for a forward pass through an MLP.

    Parameters
    ----------
    layer_sizes : list[int]  e.g. [784, 256, 128, 10]
    batch_size  : int

    Returns
    -------
    int — total FLOPs for one forward pass of the given batch size
    """
    total = 0
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        total += flops_linear(batch_size, n_in, n_out)
    return total


def flops_per_sample(layer_sizes: list) -> int:
    """FLOPs for a single sample through the MLP (batch_size=1)."""
    return count_model_flops(layer_sizes, batch_size=1)


def arithmetic_intensity_linear(batch: int, in_f: int, out_f: int,
                                 dtype=np.float32) -> float:
    """
    Arithmetic intensity of a linear layer = FLOPs / bytes read+written.

    This is the x-axis of the roofline model.

    Bytes:
      - Read weight matrix:  in_f * out_f * itemsize
      - Read input:          batch * in_f * itemsize
      - Write output:        batch * out_f * itemsize

    Parameters
    ----------
    batch, in_f, out_f : ints
    dtype              : numpy dtype for itemsize calculation

    Returns
    -------
    float — FLOPs per byte (higher = more compute-bound)
    """
    itemsize = np.dtype(dtype).itemsize
    flops    = flops_linear(batch, in_f, out_f)
    bytes_io = (in_f * out_f + batch * in_f + batch * out_f) * itemsize
    return flops / bytes_io


def format_flops(n: int) -> str:
    """Human-readable FLOPs string."""
    if n >= 1e12:
        return f"{n/1e12:.2f} TFLOPs"
    if n >= 1e9:
        return f"{n/1e9:.2f} GFLOPs"
    if n >= 1e6:
        return f"{n/1e6:.2f} MFLOPs"
    if n >= 1e3:
        return f"{n/1e3:.2f} KFLOPs"
    return f"{n} FLOPs"


# ---------------------------------------------------------------------------
# Full model breakdown report
# ---------------------------------------------------------------------------

def report_model_flops(layer_sizes: list, batch_size: int = 32,
                        dtype=np.float32):
    """
    Print a per-layer FLOPs breakdown and arithmetic intensity table.
    """
    print(f"\n{'=' * 65}")
    print(f"  FLOPs Report — MLP{layer_sizes}  (batch={batch_size})")
    print(f"{'=' * 65}")
    print(f"  {'Layer':<15} {'Shape':<20} {'FLOPs':>12} {'Arith.Int.':>12}")
    print(f"  {'-' * 61}")

    total = 0
    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        f   = flops_linear(batch_size, n_in, n_out)
        ai  = arithmetic_intensity_linear(batch_size, n_in, n_out, dtype)
        total += f
        print(f"  {'layer' + str(i+1):<15} "
              f"{str((n_in, n_out)):<20} "
              f"{format_flops(f):>12} "
              f"{ai:>11.1f}x")

    print(f"  {'=' * 61}")
    print(f"  {'TOTAL':<35} {format_flops(total):>12}")
    print(f"  {'Per sample':<35} {format_flops(total // batch_size):>12}")
    print(f"{'=' * 65}\n")

    return total


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Linear layer test
    f = flops_linear(32, 784, 256)
    print(f"Linear(32, 784→256): {format_flops(f)}  (expected ~12.8 MFLOPs)")

    # Conv layer test (ResNet-style: 64 3x3 filters, 28x28 output, 3 input channels)
    f = flops_conv(32, 3, 64, 3, 28, 28)
    print(f"Conv(32, 3→64, k=3, 28x28): {format_flops(f)}")

    # Attention test (BERT-base: 512 seq, 768 dim, 12 heads)
    f = flops_attention(32, 512, 768, 12)
    print(f"Attention(32, seq=512, d=768, 12h): {format_flops(f)}")

    # Full MLP report
    report_model_flops([784, 256, 128, 10], batch_size=32)

    # Arithmetic intensity
    ai = arithmetic_intensity_linear(32, 784, 256)
    print(f"Arithmetic intensity (32, 784→256): {ai:.1f} FLOPs/byte")
    print("(A100 GPU ridge point ~200 FLOPs/byte — this layer is memory-bound)")
