"""
worker.py — Per-worker training logic for data-parallel distributed training.

Each worker:
  1. Holds a copy of the full model
  2. Processes its local data shard
  3. Computes gradients
  4. All-reduces gradients across all workers
  5. Updates its local model (identical updates → models stay in sync)
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from distributed import DistributedWorker, CommContext


# ---------------------------------------------------------------------------
# Minimal MLP with manual backward pass
# ---------------------------------------------------------------------------

class DistributedMLP:
    """
    MLP suitable for distributed training — supports manual gradient computation.
    """

    def __init__(self, layer_sizes: list, rng: np.random.Generator):
        self.layer_sizes = layer_sizes
        self.weights: list[np.ndarray] = []
        self.biases:  list[np.ndarray] = []
        # Gradient buffers
        self.dW: list[np.ndarray] = []
        self.db: list[np.ndarray] = []

        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(6.0 / (n_in + n_out))
            W = rng.uniform(-limit, limit, (n_out, n_in)).astype(np.float64)
            b = np.zeros(n_out, dtype=np.float64)
            self.weights.append(W)
            self.biases.append(b)
            self.dW.append(np.zeros_like(W))
            self.db.append(np.zeros_like(b))

    def forward_backward(self, x: np.ndarray, y_true: np.ndarray):
        """
        Combined forward + backward pass.
        Returns scalar cross-entropy loss.
        Accumulates gradients into self.dW, self.db.
        """
        x = x.astype(np.float64)
        n = x.shape[0]

        # --- Forward ---
        activations = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W.T + b
            if i < len(self.weights) - 1:
                h = np.maximum(z, 0)       # ReLU
            else:
                # Softmax (numerically stable)
                z_shift = z - z.max(axis=1, keepdims=True)
                e = np.exp(z_shift)
                h = e / e.sum(axis=1, keepdims=True)
            activations.append(h)

        # --- Loss: cross-entropy ---
        probs = activations[-1]
        loss = -np.mean(np.log(probs[np.arange(n), y_true] + 1e-9))

        # --- Backward ---
        delta = probs.copy()
        delta[np.arange(n), y_true] -= 1
        delta /= n

        for i in reversed(range(len(self.weights))):
            h_in = activations[i]
            self.dW[i] = delta.T @ h_in
            self.db[i] = delta.sum(axis=0)
            if i > 0:
                delta = delta @ self.weights[i]
                delta *= (activations[i] > 0).astype(np.float64)   # ReLU gradient

        return float(loss)

    def zero_grad(self):
        for i in range(len(self.weights)):
            self.dW[i].fill(0.0)
            self.db[i].fill(0.0)

    def sgd_update(self, lr: float):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * self.dW[i]
            self.biases[i]  -= lr * self.db[i]

    def get_params_flat(self) -> np.ndarray:
        parts = []
        for W, b in zip(self.weights, self.biases):
            parts.extend([W.flatten(), b.flatten()])
        return np.concatenate(parts)

    def set_params_flat(self, flat: np.ndarray):
        idx = 0
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            nW = W.size
            self.weights[i] = flat[idx:idx+nW].reshape(W.shape)
            idx += nW
            nb = b.size
            self.biases[i] = flat[idx:idx+nb].reshape(b.shape)
            idx += nb


# ---------------------------------------------------------------------------
# Worker process function
# ---------------------------------------------------------------------------

def worker_process(
    rank: int,
    world_size: int,
    ctx: CommContext,
    x_shard: np.ndarray,
    y_shard: np.ndarray,
    layer_sizes: list,
    n_epochs: int,
    lr: float,
    batch_size: int,
    result_queue,     # mp.Queue to report results
    use_ring: bool = False,
):
    """
    The main function for each worker process.

    Key invariant: after every gradient all-reduce, all workers have the
    same gradients, so SGD updates are identical → models stay in sync.
    """
    rng = np.random.default_rng(42)   # Same seed → same initial weights
    model = DistributedMLP(layer_sizes, rng)
    dist  = DistributedWorker(rank, world_size, ctx)

    # Broadcast initial weights from rank 0 to ensure all workers start identically
    params = model.get_params_flat()
    params = dist.broadcast(params, src=0)
    model.set_params_flat(params)

    dist.barrier()   # ensure all workers have initialized

    step_times   = []
    comm_times   = []
    losses_per_epoch = []
    n_local = x_shard.shape[0]

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches  = 0
        perm = rng.permutation(n_local)

        for start in range(0, n_local, batch_size):
            t_step = time.perf_counter()

            idx = perm[start:start + batch_size]
            xb  = x_shard[idx]
            yb  = y_shard[idx]

            model.zero_grad()
            loss = model.forward_backward(xb, yb)
            epoch_loss += loss
            n_batches  += 1

            # All-reduce gradients: average across all workers
            t_comm = time.perf_counter()
            for i in range(len(model.weights)):
                if use_ring:
                    model.dW[i] = dist.ring_all_reduce(model.dW[i]) / world_size
                    model.db[i] = dist.ring_all_reduce(model.db[i]) / world_size
                else:
                    model.dW[i] = dist.all_reduce(model.dW[i], op='mean')
                    model.db[i] = dist.all_reduce(model.db[i], op='mean')
            comm_times.append(time.perf_counter() - t_comm)

            model.sgd_update(lr)
            step_times.append(time.perf_counter() - t_step)

        losses_per_epoch.append(epoch_loss / max(n_batches, 1))

    dist.barrier()

    result_queue.put({
        "rank"           : rank,
        "final_loss"     : losses_per_epoch[-1],
        "losses"         : losses_per_epoch,
        "mean_step_ms"   : float(np.mean(step_times)) * 1000,
        "mean_comm_ms"   : float(np.mean(comm_times)) * 1000,
        "comm_fraction"  : float(np.mean(comm_times)) / (float(np.mean(step_times)) + 1e-9),
        "final_params"   : model.get_params_flat(),
    })
