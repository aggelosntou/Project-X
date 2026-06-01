"""
benchmark_pipeline.py — Measure DataLoader throughput with varying num_workers.

Benchmark setup:
  - 10 000 synthetic samples, each a (100,) float32 feature vector
  - batch_size = 64
  - num_workers = 0, 1, 2, 4
  - Measure: samples/second and total epoch time

Run:
    python src/benchmark_pipeline.py
"""

import sys
import os
import time
import numpy as np

# Must use if __name__ guard for multiprocessing on macOS (spawn start method)
sys.path.insert(0, os.path.dirname(__file__))

from dataset    import ArrayDataset
from dataloader import DataLoader
from transforms import Normalize, AddGaussianNoise, Compose, ToFloat32


def run_epoch(loader):
    """Iterate through a full epoch and return (total_samples, elapsed_s)."""
    t0 = time.perf_counter()
    total = 0
    for xb, yb in loader:
        total += len(xb)
        # Simulate a trivial compute step (sum) — keeps the loop realistic
        _ = xb.sum()
    elapsed = time.perf_counter() - t0
    return total, elapsed


def benchmark(dataset, batch_size=64, n_epochs=1, worker_counts=(0, 1, 2, 4),
              transform=None):
    """
    Run the benchmark for each num_workers value.

    Returns list of dicts with results.
    """
    results = []

    for nw in worker_counts:
        # Warm-up epoch to absorb process spawn startup cost
        # (not counted in the timing)
        warmup_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=nw, prefetch_factor=2, transform=transform
        )
        # Only warm up for num_workers > 0 (spawn cost is significant)
        if nw > 0:
            for _ in warmup_loader:
                pass

        # Timed epoch(s)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=nw, prefetch_factor=2, transform=transform
        )

        elapsed_all = 0.0
        for _ in range(n_epochs):
            total_samples, elapsed = run_epoch(loader)
            elapsed_all += elapsed

        avg_elapsed = elapsed_all / n_epochs
        samples_per_sec = total_samples / avg_elapsed

        results.append({
            "num_workers"    : nw,
            "total_samples"  : total_samples,
            "epoch_time_s"   : avg_elapsed,
            "samples_per_sec": samples_per_sec,
        })

        print(f"  num_workers={nw:2d}  "
              f"epoch_time={avg_elapsed:.3f}s  "
              f"samples/sec={samples_per_sec:,.0f}")

    return results


def print_table(results, baseline_idx=0):
    """Print formatted results table."""
    baseline_sps = results[baseline_idx]["samples_per_sec"]

    print()
    print(f"{'=' * 65}")
    print(f"  DataLoader Throughput Benchmark")
    print(f"{'=' * 65}")
    print(f"  {'num_workers':>12} {'samples/sec':>14} {'epoch_time':>12} {'speedup':>9}")
    print(f"  {'-' * 53}")
    for r in results:
        speedup = r["samples_per_sec"] / baseline_sps
        print(f"  {r['num_workers']:>12} "
              f"{r['samples_per_sec']:>14,.0f} "
              f"{r['epoch_time_s']:>11.3f}s "
              f"{speedup:>8.2f}x")
    print(f"{'=' * 65}")


def explain_results(results):
    """
    Print educational commentary based on the benchmark results.
    The commentary automatically adapts to what the numbers show.
    """
    print()
    print("Observations:")
    baseline = results[0]
    best = max(results, key=lambda r: r["samples_per_sec"])

    print(f"  Baseline (num_workers=0): {baseline['samples_per_sec']:,.0f} samples/sec")
    print(f"  Best ({best['num_workers']} workers):      {best['samples_per_sec']:,.0f} samples/sec")

    if best["num_workers"] == 0:
        print()
        print("  For this synthetic dataset (pure in-memory NumPy), num_workers=0")
        print("  is fastest. This is expected:")
        print("  - Multiprocessing has spawn overhead (~10-50ms per process on macOS)")
        print("  - The dataset fits in memory, so there is no IO latency to hide")
        print("  - Process serialisation (pickle) of numpy arrays adds latency")
    else:
        speedup = best["samples_per_sec"] / baseline["samples_per_sec"]
        print(f"  Speedup from workers: {speedup:.2f}x")
        print()
        print("  Workers help here because the transform pipeline adds CPU cost")
        print("  that can be parallelised across processes.")

    print()
    print("Why num_workers > 0 can be SLOWER for small in-memory datasets:")
    print("  1. Process spawn overhead (macOS uses 'spawn' start method,")
    print("     which reimports all modules — ~20-100ms per process)")
    print("  2. Data must be serialised (pickle) to cross process boundaries")
    print("  3. Result queue has synchronisation overhead")
    print("  4. The worker process has no work to overlap (data is already in RAM)")
    print()
    print("Why num_workers > 0 helps for real datasets:")
    print("  1. Disk IO is slow (reading images from SSD: ~10ms per image)")
    print("  2. Transforms (resize, augment) are CPU-bound and parallelisable")
    print("  3. While GPU computes on batch N, CPU workers prepare batch N+1 (prefetch)")
    print("  4. With persistent workers (PyTorch style), spawn overhead is amortised")
    print()
    print("Rule of thumb: use num_workers = number of CPU cores / 2 for image datasets")
    print("               use num_workers = 0 for small in-memory datasets")


def main():
    # -----------------------------------------------------------------------
    # Dataset: 10 000 samples, each (100,) feature vector
    # -----------------------------------------------------------------------
    print("Building dataset: 10 000 samples × (100,) features ...")
    rng = np.random.default_rng(42)
    X   = rng.standard_normal((10_000, 100)).astype(np.float32)
    y   = rng.integers(0, 10, 10_000)
    dataset = ArrayDataset(X, y)

    BATCH_SIZE = 64

    # -----------------------------------------------------------------------
    # Benchmark 1: No transform (baseline — pure indexing cost)
    # -----------------------------------------------------------------------
    print("\n--- Benchmark 1: No transform (pure array indexing) ---")
    results_no_transform = benchmark(
        dataset,
        batch_size=BATCH_SIZE,
        n_epochs=2,
        worker_counts=[0, 1, 2, 4],
        transform=None
    )
    print_table(results_no_transform)

    # -----------------------------------------------------------------------
    # Benchmark 2: With a transform pipeline (adds CPU cost per sample)
    # -----------------------------------------------------------------------
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    transform = Compose([
        ToFloat32(),
        Normalize(mean=mean, std=std),
        AddGaussianNoise(std=0.05),
    ])

    print("\n--- Benchmark 2: With Normalize + AddGaussianNoise transform ---")
    results_transform = benchmark(
        dataset,
        batch_size=BATCH_SIZE,
        n_epochs=2,
        worker_counts=[0, 1, 2, 4],
        transform=transform
    )
    print_table(results_transform)

    explain_results(results_no_transform)

    print()
    print("Threading vs multiprocessing summary:")
    print("  - threading: shared memory (fast IPC), but Python's GIL prevents")
    print("    true CPU parallelism for Python code (only IO-bound tasks benefit)")
    print("  - multiprocessing: separate memory spaces (slow IPC via pickle),")
    print("    each process has its own GIL, enabling true CPU parallelism")
    print("  - For NumPy-heavy transforms, multiprocessing wins because NumPy")
    print("    releases the GIL during C-level computations.")
    print()


if __name__ == "__main__":
    # Required for multiprocessing on macOS (spawn start method)
    mp_ctx = __import__('multiprocessing')
    mp_ctx.set_start_method('spawn', force=True)
    main()
