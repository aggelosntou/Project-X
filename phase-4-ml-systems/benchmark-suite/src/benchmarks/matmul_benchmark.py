"""
matmul_benchmark.py
===================
Benchmark NumPy square matrix multiplication (GEMM) at various sizes.

Measures peak GFLOPS to determine whether NumPy is using an optimised
BLAS backend (OpenBLAS, MKL, Accelerate) on your machine.

A standard matrix multiply A @ B where A, B are (N x N):
  FLOPs = 2 * N^3    (N^2 dot products of length N)
"""

import numpy as np
import time
import sys
import os

# Allow direct execution from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def benchmark_matmul(n_runs: int = 10, warmup: int = 3) -> dict:
    """
    Run square GEMM benchmark.

    Returns a dict with keys 'status', 'rows' (list of per-size results),
    and 'peak_gflops'.
    """
    print("\n=== Matrix Multiply Benchmark ===")
    print(f"{'N':>6}  {'GFLOPS':>12}  {'time_ms':>10}  {'dtype':>8}")
    print("-" * 42)

    sizes = [64, 128, 256, 512, 1024, 2048]
    rows = []
    peak_gflops = 0.0

    rng = np.random.default_rng(0)

    for N in sizes:
        A = rng.standard_normal((N, N)).astype(np.float32)
        B = rng.standard_normal((N, N)).astype(np.float32)

        # Warmup: ensure JIT / library initialisation is done
        for _ in range(warmup):
            _ = A @ B

        # Timed runs — take the minimum (best-case = no OS jitter)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            C = A @ B  # noqa: F841
            times.append(time.perf_counter() - t0)

        t_min = min(times)
        t_mean = sum(times) / len(times)
        t_std  = (sum((x - t_mean) ** 2 for x in times) / len(times)) ** 0.5

        flops = 2 * N ** 3
        gflops = flops / t_min / 1e9
        if gflops > peak_gflops:
            peak_gflops = gflops

        row = {
            'N':          N,
            'gflops':     round(gflops,     2),
            'time_ms':    round(t_min * 1000, 3),
            'mean_ms':    round(t_mean * 1000, 3),
            'std_ms':     round(t_std  * 1000, 3),
        }
        rows.append(row)
        print(f"{N:>6}  {gflops:>12.2f}  {t_min * 1000:>10.3f}  float32")

    print(f"\nPeak: {peak_gflops:.2f} GFLOPS (float32)")

    return {
        'status':      'completed',
        'rows':        rows,
        'peak_gflops': round(peak_gflops, 2),
    }


if __name__ == '__main__':
    benchmark_matmul()
