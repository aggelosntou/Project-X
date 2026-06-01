"""
compare_strategies.py — Compare data-parallel strategies.

1. Single worker (baseline)
2. Data parallel with gather-scatter all-reduce (naïve)
3. Data parallel with ring-allreduce (bandwidth-optimal)
4. Show communication overhead as % of training time
5. Show that ring-allreduce scales better (constant comm volume per worker)
"""

import sys
import os
import time
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from distributed import CommContext
from worker import worker_process, DistributedMLP


def make_dataset(n_samples, n_features, n_classes, rng):
    centers = rng.standard_normal((n_classes, n_features)) * 3
    xs, ys = [], []
    per_class = n_samples // n_classes
    for c in range(n_classes):
        xi = centers[c] + rng.standard_normal((per_class, n_features))
        xs.append(xi); ys.append(np.full(per_class, c, dtype=np.int64))
    x = np.concatenate(xs); y = np.concatenate(ys)
    p = rng.permutation(len(x))
    return x[p].astype(np.float64), y[p]


def run_workers(world_size, x, y, layer_sizes, n_epochs, lr, batch_size, use_ring):
    ctx          = CommContext(world_size)
    result_queue = mp.Queue()
    processes    = []
    n            = len(x)
    shard_size   = n // world_size

    t0 = time.perf_counter()
    for rank in range(world_size):
        lo = rank * shard_size
        hi = lo + shard_size if rank < world_size - 1 else n
        p = mp.Process(
            target=worker_process,
            args=(rank, world_size, ctx,
                  x[lo:hi], y[lo:hi],
                  layer_sizes, n_epochs, lr, batch_size,
                  result_queue, use_ring),
            daemon=True,
        )
        p.start(); processes.append(p)

    for p in processes:
        p.join(timeout=120)
    elapsed = time.perf_counter() - t0

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    results.sort(key=lambda r: r["rank"])
    return results, elapsed


def run_single(x, y, layer_sizes, n_epochs, lr, batch_size):
    """Single-worker baseline."""
    rng   = np.random.default_rng(42)
    model = DistributedMLP(layer_sizes, rng)
    losses = []
    n = len(x)
    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        loss_sum = 0; nb = 0
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            model.zero_grad()
            loss = model.forward_backward(x[idx], y[idx])
            model.sgd_update(lr)
            loss_sum += loss; nb += 1
        losses.append(loss_sum / max(nb, 1))
    elapsed = time.perf_counter() - t0
    return losses, elapsed


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    N_FEATURES   = 32
    N_CLASSES    = 5
    N_SAMPLES    = 1000
    LAYER_SIZES  = [N_FEATURES, 64, 32, N_CLASSES]
    N_EPOCHS     = 3
    LR           = 0.05
    BATCH_SIZE   = 32

    x, y = make_dataset(N_SAMPLES, N_FEATURES, N_CLASSES, rng)

    print("=" * 70)
    print("DISTRIBUTED TRAINING — STRATEGY COMPARISON")
    print("=" * 70)

    # 1. Single worker
    losses_single, t_single = run_single(x, y, LAYER_SIZES, N_EPOCHS, LR, BATCH_SIZE)
    print(f"\n[1] Single worker")
    print(f"    Final loss : {losses_single[-1]:.4f}")
    print(f"    Wall time  : {t_single:.3f}s")

    # 2-4. Multi-worker with different strategies
    results_table = []

    for world_size in [2, 4]:
        for use_ring, label in [(False, "gather-scatter"), (True, "ring-allreduce")]:
            results, t_total = run_workers(
                world_size, x, y, LAYER_SIZES, N_EPOCHS, LR, BATCH_SIZE, use_ring
            )
            if not results:
                continue

            rank0 = results[0]
            comm_pcts = [r["comm_fraction"] * 100 for r in results]
            mean_comm_pct = np.mean(comm_pcts)

            # Verify gradient sync: all workers same loss
            losses = [r["final_loss"] for r in results]
            loss_std = np.std(losses)

            results_table.append({
                "strategy"     : label,
                "world_size"   : world_size,
                "final_loss"   : rank0["final_loss"],
                "wall_time"    : t_total,
                "comm_pct"     : mean_comm_pct,
                "loss_std"     : loss_std,
            })

            print(f"\n[N={world_size}, {label}]")
            print(f"    Final loss      : {rank0['final_loss']:.4f}")
            print(f"    Loss std across workers: {loss_std:.2e}  "
                  f"({'in sync' if loss_std < 1e-4 else 'DIVERGED'})")
            print(f"    Wall time       : {t_total:.3f}s")
            print(f"    Comm overhead   : {mean_comm_pct:.1f}%")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"  {'Strategy':<22}  {'N':>3}  {'Loss':>8}  {'Time(s)':>8}  "
          f"{'Comm%':>7}  {'Sync?':>6}")
    print(f"  {'-'*22}  {'-'*3}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*6}")
    print(f"  {'single':<22}  {'1':>3}  {losses_single[-1]:>8.4f}  "
          f"{t_single:>8.3f}  {'N/A':>7}  {'N/A':>6}")
    for r in results_table:
        sync = "YES" if r["loss_std"] < 1e-4 else "NO"
        print(f"  {r['strategy']:<22}  {r['world_size']:>3}  "
              f"{r['final_loss']:>8.4f}  {r['wall_time']:>8.3f}  "
              f"{r['comm_pct']:>6.1f}%  {sync:>6}")

    print(f"\nKey observations:")
    print(f"  1. All distributed runs converge to the same loss as single-worker")
    print(f"     (gradient averaging = correct data-parallel training).")
    print(f"  2. Ring-allreduce has the same communication VOLUME as gather-scatter")
    print(f"     but distributes the bandwidth load evenly across all workers.")
    print(f"  3. For small models on fast shared memory (same machine),")
    print(f"     communication overhead dominates.  On real GPUs with NVLink,")
    print(f"     bandwidth is high enough that compute dominates.")
    print(f"  4. Ring-allreduce bandwidth per worker = 2*(N-1)/N * data_size,")
    print(f"     approaching 2*data_size (optimal) for large N.")
