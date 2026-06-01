"""
attention_benchmark.py
======================
Pure NumPy implementation of scaled dot-product multi-head attention,
benchmarked across sequence lengths.

No CUDA required — runs on any machine.

Attention equation (per head):
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

FLOPs breakdown (batch=1, h heads, seq N, d_head d):
    QK^T:    2 * h * N^2 * d
    softmax: ~5 * h * N^2   (max + exp + sum + divide — approx)
    AV:      2 * h * N^2 * d
    Total:   4 * h * N^2 * d + 5 * h * N^2
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
) -> np.ndarray:
    """
    Compute attention output.

    Args:
        Q, K, V: arrays of shape (batch, heads, seq_len, d_head)

    Returns:
        Output of shape (batch, heads, seq_len, d_head)
    """
    d_k = Q.shape[-1]
    scale = 1.0 / np.sqrt(d_k)

    # scores: (batch, heads, seq_len, seq_len)
    scores = Q @ K.transpose(0, 1, 3, 2) * scale

    # Numerically stable softmax along last axis
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)

    # Output: (batch, heads, seq_len, d_head)
    return weights @ V


def benchmark_attention(n_runs: int = 5, warmup: int = 2) -> dict:
    """
    Benchmark attention for seq_len in [32, 64, 128, 256, 512].

    Returns dict with 'status' and 'rows'.
    """
    print("\n=== Attention Benchmark ===")
    print(f"{'seq_len':>8}  {'time_ms':>10}  {'mem_MB':>8}  {'GFLOPS':>8}")
    print("-" * 42)

    batch  = 1
    heads  = 8
    d_head = 64

    seq_lens = [32, 64, 128, 256, 512]
    rows = []
    rng  = np.random.default_rng(1)

    for seq_len in seq_lens:
        Q = rng.standard_normal((batch, heads, seq_len, d_head)).astype(np.float32)
        K = rng.standard_normal((batch, heads, seq_len, d_head)).astype(np.float32)
        V = rng.standard_normal((batch, heads, seq_len, d_head)).astype(np.float32)

        # Approximate peak memory: Q + K + V + scores + output  (float32)
        qkv_elems   = batch * heads * seq_len * d_head
        scr_elems   = batch * heads * seq_len * seq_len
        mem_mb = (3 * qkv_elems + scr_elems + qkv_elems) * 4 / 1024 ** 2

        # Warmup
        for _ in range(warmup):
            scaled_dot_product_attention(Q, K, V)

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            scaled_dot_product_attention(Q, K, V)
            times.append(time.perf_counter() - t0)

        t_min  = min(times)
        t_mean = sum(times) / len(times)

        # FLOPs: QK^T + softmax (approx) + AV
        flops = (4 * batch * heads * seq_len ** 2 * d_head
                 + 5 * batch * heads * seq_len ** 2)
        gflops = flops / t_min / 1e9

        row = {
            'seq_len':   seq_len,
            'time_ms':   round(t_min  * 1000, 3),
            'mean_ms':   round(t_mean * 1000, 3),
            'mem_mb':    round(mem_mb, 2),
            'gflops':    round(gflops, 4),
        }
        rows.append(row)
        print(f"{seq_len:>8}  {t_min*1000:>10.3f}  {mem_mb:>8.1f}  {gflops:>8.4f}")

    return {'status': 'completed', 'rows': rows}


if __name__ == '__main__':
    benchmark_attention()
