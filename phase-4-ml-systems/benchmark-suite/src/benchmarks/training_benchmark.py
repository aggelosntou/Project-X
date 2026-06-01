"""
training_benchmark.py
=====================
Benchmark a complete training step on a small MLP using pure NumPy.

Phases timed:
    1. Forward pass
    2. Backward pass (manual backprop)
    3. Weight update (SGD with momentum)

The backward pass is a hand-written implementation to make the NumPy
compute cost visible (no autograd framework).

MLP architecture:
    input(128) -> Linear(256) -> ReLU -> Linear(128) -> ReLU -> Linear(10)
    Loss: mean squared error (simple, avoids softmax complexity)
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ---------------------------------------------------------------------------
# Minimal MLP with manual forward + backward
# ---------------------------------------------------------------------------

class Layer:
    """Single linear layer with optional ReLU."""

    def __init__(self, in_dim: int, out_dim: int, rng):
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.W  = rng.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float32)
        self.b  = np.zeros(out_dim, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        # Momentum buffers
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

    def forward(self, x: np.ndarray, relu: bool = True) -> np.ndarray:
        self._x  = x                             # cache for backward
        self._z  = x @ self.W + self.b           # pre-activation
        self._relu = relu
        if relu:
            self._mask = self._z > 0
            return self._z * self._mask
        return self._z

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        if self._relu:
            d_out = d_out * self._mask           # ReLU gradient
        self.dW = self._x.T @ d_out / d_out.shape[0]
        self.db = d_out.mean(axis=0)
        return d_out @ self.W.T                  # gradient w.r.t. input

    def update(self, lr: float = 0.01, momentum: float = 0.9) -> None:
        self.vW = momentum * self.vW - lr * self.dW
        self.vb = momentum * self.vb - lr * self.db
        self.W += self.vW
        self.b += self.vb


class SmallMLP:
    def __init__(self):
        rng = np.random.default_rng(42)
        self.layers = [
            Layer(128, 256, rng),
            Layer(256, 128, rng),
            Layer(128,  10, rng),
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i, layer in enumerate(self.layers):
            x = layer.forward(x, relu=(i < len(self.layers) - 1))
        return x

    def backward(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, lr: float = 0.01) -> None:
        for layer in self.layers:
            layer.update(lr)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_training(
    batch_size: int = 32,
    n_steps: int    = 100,
    warmup: int     = 5,
) -> dict:
    """
    Benchmark one training step broken into: forward, backward, update.

    Returns dict with 'status', 'batch_size', 'phase_results'.
    """
    print(f"\n=== Training Step Benchmark (batch_size={batch_size}, n_steps={n_steps}) ===")

    rng   = np.random.default_rng(7)
    model = SmallMLP()

    x_all = rng.standard_normal((n_steps + warmup, batch_size, 128)).astype(np.float32)
    y_all = rng.standard_normal((n_steps + warmup, batch_size,  10)).astype(np.float32)

    # Warmup
    for s in range(warmup):
        x, y   = x_all[s], y_all[s]
        pred   = model.forward(x)
        grad   = 2.0 * (pred - y) / batch_size   # dL/d_logits for MSE
        model.backward(grad)
        model.update()

    # Timed runs — measure each phase
    fwd_times  = []
    bwd_times  = []
    upd_times  = []

    for s in range(n_steps):
        x, y = x_all[warmup + s], y_all[warmup + s]

        # Forward
        t0   = time.perf_counter()
        pred = model.forward(x)
        fwd_times.append(time.perf_counter() - t0)

        # Compute gradient (included in backward timing)
        grad = 2.0 * (pred - y) / batch_size

        # Backward
        t0 = time.perf_counter()
        model.backward(grad)
        bwd_times.append(time.perf_counter() - t0)

        # Update
        t0 = time.perf_counter()
        model.update()
        upd_times.append(time.perf_counter() - t0)

    def stats(ts):
        mean = sum(ts) / len(ts)
        std  = (sum((t - mean) ** 2 for t in ts) / len(ts)) ** 0.5
        return mean, std, min(ts)

    fwd_mean, fwd_std, fwd_min = stats(fwd_times)
    bwd_mean, bwd_std, bwd_min = stats(bwd_times)
    upd_mean, upd_std, upd_min = stats(upd_times)

    total_mean = fwd_mean + bwd_mean + upd_mean

    print(f"\n{'Phase':<12} {'mean_ms':>10} {'std_ms':>10} {'min_ms':>10} {'% total':>10}")
    print("-" * 56)
    for name, mean_t, std_t, min_t in [
        ('forward',  fwd_mean, fwd_std, fwd_min),
        ('backward', bwd_mean, bwd_std, bwd_min),
        ('update',   upd_mean, upd_std, upd_min),
    ]:
        pct = 100.0 * mean_t / total_mean if total_mean > 0 else 0
        print(f"{name:<12} {mean_t*1000:>10.4f} {std_t*1000:>10.4f} "
              f"{min_t*1000:>10.4f} {pct:>9.1f}%")

    print(f"\n  Total step: {total_mean * 1000:.4f} ms  "
          f"({1.0 / total_mean:.1f} steps/sec)")

    phase_results = {
        'forward':  {'mean_ms': round(fwd_mean * 1000, 4),
                     'std_ms':  round(fwd_std  * 1000, 4),
                     'pct':     round(100 * fwd_mean / total_mean, 2)},
        'backward': {'mean_ms': round(bwd_mean * 1000, 4),
                     'std_ms':  round(bwd_std  * 1000, 4),
                     'pct':     round(100 * bwd_mean / total_mean, 2)},
        'update':   {'mean_ms': round(upd_mean * 1000, 4),
                     'std_ms':  round(upd_std  * 1000, 4),
                     'pct':     round(100 * upd_mean / total_mean, 2)},
    }

    return {
        'status':        'completed',
        'batch_size':    batch_size,
        'n_steps':       n_steps,
        'total_step_ms': round(total_mean * 1000, 4),
        'steps_per_sec': round(1.0 / total_mean, 1),
        'phase_results': phase_results,
    }


if __name__ == '__main__':
    benchmark_training()
