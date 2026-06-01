"""
train_distributed.py — Launch N workers for data-parallel distributed training.

Spawns N processes (simulating N GPUs).  Each gets a unique data shard.
After each step, gradients are all-reduced so all workers stay in sync.

Key verification: the final loss with N workers should equal the loss
with 1 worker (same effective gradient = average over all data shards).

Usage:
    python3 src/train_distributed.py
    python3 src/train_distributed.py --world-size 4 --epochs 10
"""

import sys
import os
import time
import argparse
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from distributed import CommContext
from worker import worker_process, DistributedMLP


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_dataset(n_samples: int, n_features: int, n_classes: int,
                 rng: np.random.Generator):
    centers = rng.standard_normal((n_classes, n_features)).astype(np.float64) * 3
    xs, ys = [], []
    per_class = n_samples // n_classes
    for c in range(n_classes):
        xi = centers[c] + rng.standard_normal((per_class, n_features)).astype(np.float64)
        xs.append(xi)
        ys.append(np.full(per_class, c, dtype=np.int64))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    perm = rng.permutation(len(x))
    return x[perm], y[perm]


# ---------------------------------------------------------------------------
# Launch N workers
# ---------------------------------------------------------------------------

def run_distributed(
    world_size: int,
    x: np.ndarray,
    y: np.ndarray,
    layer_sizes: list,
    n_epochs: int = 5,
    lr: float = 0.01,
    batch_size: int = 32,
    use_ring: bool = False,
):
    """
    Spawn world_size processes for data-parallel training.
    Returns list of per-worker result dicts.
    """
    ctx         = CommContext(world_size)
    result_queue = mp.Queue()
    processes   = []

    # Partition data: each worker gets a contiguous shard
    n          = len(x)
    shard_size = n // world_size

    t_start = time.perf_counter()

    for rank in range(world_size):
        lo = rank * shard_size
        hi = lo + shard_size if rank < world_size - 1 else n
        x_shard = x[lo:hi]
        y_shard = y[lo:hi]

        p = mp.Process(
            target=worker_process,
            args=(rank, world_size, ctx,
                  x_shard, y_shard,
                  layer_sizes, n_epochs, lr, batch_size,
                  result_queue, use_ring),
            daemon=True,
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join(timeout=120)

    elapsed = time.perf_counter() - t_start

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    results.sort(key=lambda r: r["rank"])

    return results, elapsed


# ---------------------------------------------------------------------------
# Reference: single-worker training
# ---------------------------------------------------------------------------

def run_single_worker(
    x: np.ndarray,
    y: np.ndarray,
    layer_sizes: list,
    n_epochs: int = 5,
    lr: float = 0.01,
    batch_size: int = 32,
):
    """Run on rank-0's data shard for comparison."""
    rng   = np.random.default_rng(42)
    model = DistributedMLP(layer_sizes, rng)
    n     = len(x)
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches  = 0
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            model.zero_grad()
            loss = model.forward_backward(x[idx], y[idx])
            model.sgd_update(lr)
            epoch_loss += loss
            n_batches  += 1
        losses.append(epoch_loss / max(n_batches, 1))

    return losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2,
                        help="Number of simulated workers (default: 2)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--ring", action="store_true",
                        help="Use ring-allreduce instead of gather-scatter")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    n_features = 32
    n_classes  = 5
    n_samples  = 800
    layer_sizes = [n_features, 64, 32, n_classes]

    print("=" * 65)
    print(f"DATA-PARALLEL DISTRIBUTED TRAINING  (world_size={args.world_size})")
    print("=" * 65)

    x, y = make_dataset(n_samples, n_features, n_classes, rng)

    # --- Distributed training ---
    mode = "ring-allreduce" if args.ring else "gather-scatter"
    print(f"\nLaunching {args.world_size} worker processes  (mode: {mode})...")
    results, elapsed = run_distributed(
        args.world_size, x, y, layer_sizes,
        n_epochs=args.epochs, lr=args.lr,
        batch_size=32, use_ring=args.ring,
    )

    if not results:
        print("ERROR: No results received from workers.")
        sys.exit(1)

    print(f"Total wall time: {elapsed:.2f}s")
    print(f"\nPer-worker results:")
    print(f"  {'Rank':>4}  {'Final Loss':>12}  {'Step ms':>10}  {'Comm ms':>10}  {'Comm %':>8}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*8}")
    for r in results:
        comm_pct = r["comm_fraction"] * 100
        print(f"  {r['rank']:>4}  {r['final_loss']:>12.6f}  "
              f"{r['mean_step_ms']:>10.3f}  "
              f"{r['mean_comm_ms']:>10.3f}  "
              f"{comm_pct:>7.1f}%")

    # Verify: all workers should have the same final loss (same gradients)
    final_losses = [r["final_loss"] for r in results]
    loss_std = float(np.std(final_losses))
    print(f"\nLoss consistency check:")
    print(f"  Mean final loss: {np.mean(final_losses):.6f}")
    print(f"  Std dev:         {loss_std:.2e}  (should be ~0 after all-reduce)")

    # Check params are identical across workers
    params_list = [r["final_params"] for r in results]
    param_diffs = [np.max(np.abs(params_list[0] - p)) for p in params_list[1:]]
    if param_diffs:
        max_diff = max(param_diffs)
        print(f"  Max param diff across workers: {max_diff:.2e}  "
              f"({'OK' if max_diff < 1e-6 else 'WARNING: workers diverged!'})")

    # --- Communication overhead ---
    mean_comm_pct = np.mean([r["comm_fraction"]*100 for r in results])
    print(f"\nCommunication overhead: {mean_comm_pct:.1f}% of total step time")
    print(f"  (This is the cost of gradient synchronisation.)")
    print(f"  (For large models on slow networks, this can be 50–90%.)")
    print(f"  (Gradient compression / asynchronous SGD can reduce this.)")

    # --- Loss trajectory ---
    print(f"\nLoss trajectory (rank 0):")
    for ep, loss in enumerate(results[0]["losses"]):
        print(f"  Epoch {ep+1}: {loss:.4f}")
