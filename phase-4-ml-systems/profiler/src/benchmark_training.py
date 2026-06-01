"""
benchmark_training.py — Profile a complete training loop.

Runs 100 steps of an MLP training procedure and produces:
  1. Profiler timing report (forward, loss, backward, optimizer, data loading)
  2. FLOPs/second throughput estimate
  3. Peak memory measurement via tracemalloc
  4. Theoretical memory breakdown (weights, activations, gradients, Adam state)

Run:
    python src/benchmark_training.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from profiler        import Profiler
from memory_profiler import track_peak_memory, model_memory_breakdown
from flops_counter   import count_model_flops, format_flops, report_model_flops


# ---------------------------------------------------------------------------
# Self-contained MLP with explicit forward / backward
# ---------------------------------------------------------------------------

class MLP:
    """MLP with separate forward and backward methods, suitable for profiling."""

    def __init__(self, layer_sizes, rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        self.layer_sizes = layer_sizes
        self.weights  = []
        self.biases   = []
        self.m_w      = []   # Adam first moment — weights
        self.m_b      = []   # Adam first moment — biases
        self.v_w      = []   # Adam second moment — weights
        self.v_b      = []   # Adam second moment — biases

        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(6.0 / (n_in + n_out))
            W = rng.uniform(-limit, limit, (n_out, n_in)).astype(np.float32)
            b = np.zeros(n_out, dtype=np.float32)
            self.weights.append(W)
            self.biases.append(b)
            self.m_w.append(np.zeros_like(W))
            self.m_b.append(np.zeros_like(b))
            self.v_w.append(np.zeros_like(W))
            self.v_b.append(np.zeros_like(b))

        self._activations = []   # stored during forward for backward

    def forward(self, x):
        """
        Forward pass, storing activations for backprop.

        Returns logits (N, n_classes).
        """
        self._activations = [x.astype(np.float32)]
        h = self._activations[0]
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W.T + b
            h = np.maximum(z, 0.0) if i < len(self.weights) - 1 else z
            self._activations.append(h)
        return self._activations[-1]

    def backward(self, delta):
        """
        Backward pass given gradient w.r.t. logits (delta).

        Stores gradients in self.dW, self.db.
        Returns nothing (gradients are stored internally for optimizer step).
        """
        self.dW = [None] * len(self.weights)
        self.db = [None] * len(self.biases)

        for i in reversed(range(len(self.weights))):
            h_in   = self._activations[i]
            self.dW[i] = delta.T @ h_in
            self.db[i] = delta.sum(axis=0)
            if i > 0:
                delta = delta @ self.weights[i]
                delta = delta * (self._activations[i] > 0).astype(np.float32)

    def optimizer_step(self, lr=0.001, beta1=0.9, beta2=0.999,
                        eps=1e-8, t=1):
        """Adam optimizer update."""
        for i in range(len(self.weights)):
            # Weights
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * self.dW[i]
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * self.dW[i] ** 2
            m_hat = self.m_w[i] / (1 - beta1 ** t)
            v_hat = self.v_w[i] / (1 - beta2 ** t)
            self.weights[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

            # Biases
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * self.db[i]
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * self.db[i] ** 2
            m_hat = self.m_b[i] / (1 - beta1 ** t)
            v_hat = self.v_b[i] / (1 - beta2 ** t)
            self.biases[i]  -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ---------------------------------------------------------------------------
# Loss utilities
# ---------------------------------------------------------------------------

def softmax(x):
    e = np.exp(x.astype(np.float64) - x.astype(np.float64).max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def cross_entropy_loss_and_grad(logits, labels):
    """
    Compute CE loss and gradient w.r.t. logits in one step.

    Returns (loss_scalar, grad_logits)
    """
    n      = len(labels)
    probs  = softmax(logits)
    loss   = -float(np.mean(np.log(probs[np.arange(n), labels] + 1e-9)))
    delta  = probs.copy()
    delta[np.arange(n), labels] -= 1.0
    delta /= n
    return loss, delta.astype(np.float32)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

N_STEPS   = 100
BATCH     = 64
IN_DIM    = 256
N_CLASSES = 10
LAYER_SIZES = [IN_DIM, 512, 256, 128, N_CLASSES]


def full_training_loop():
    """
    Run N_STEPS training steps and return (profiler, losses).
    Called inside track_peak_memory to capture peak RAM.
    """
    rng   = np.random.default_rng(42)
    model = MLP(LAYER_SIZES, rng=rng)
    prof  = Profiler()
    losses = []

    with prof:
        for step in range(N_STEPS):

            # ----------------------------------------------------------
            # Phase 1: Data loading
            # ----------------------------------------------------------
            with prof.timer('data_loading'):
                X = rng.standard_normal((BATCH, IN_DIM)).astype(np.float32)
                y = rng.integers(0, N_CLASSES, BATCH)

            # ----------------------------------------------------------
            # Phase 2: Forward pass
            # ----------------------------------------------------------
            with prof.timer('forward'):
                logits = model.forward(X)

            # ----------------------------------------------------------
            # Phase 3: Loss computation
            # ----------------------------------------------------------
            with prof.timer('loss_computation'):
                loss, delta = cross_entropy_loss_and_grad(logits, y)
                losses.append(loss)

            # ----------------------------------------------------------
            # Phase 4: Backward pass
            # ----------------------------------------------------------
            with prof.timer('backward'):
                model.backward(delta)

            # ----------------------------------------------------------
            # Phase 5: Optimizer step
            # ----------------------------------------------------------
            with prof.timer('optimizer_step'):
                model.optimizer_step(lr=0.001, t=step + 1)

    return prof, losses, model


def main():
    print("=" * 65)
    print("  Training Loop Profiler Benchmark")
    print(f"  Architecture: {LAYER_SIZES}")
    print(f"  Batch size: {BATCH}  |  Steps: {N_STEPS}")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # 1. Run with timing
    # -----------------------------------------------------------------------
    print("\nRunning benchmark (100 steps) ...")
    t_start = time.perf_counter()
    (prof, losses, model), peak_mb = track_peak_memory(full_training_loop)
    total_wall = time.perf_counter() - t_start

    # -----------------------------------------------------------------------
    # 2. Profiler report
    # -----------------------------------------------------------------------
    prof.report("Training Loop Phases — 100 steps")

    # -----------------------------------------------------------------------
    # 3. FLOPs analysis
    # -----------------------------------------------------------------------
    total_flops = count_model_flops(LAYER_SIZES, batch_size=BATCH)
    # Backward pass is approximately 2x forward (compute grad_input + grad_weight)
    flops_per_step = total_flops * 3   # forward + backward (≈ 2x forward)

    fwd_time_s = prof.total_time('forward') / N_STEPS
    bwd_time_s = prof.total_time('backward') / N_STEPS
    total_compute_s = prof.total_time('forward') + prof.total_time('backward')

    gflops_per_sec = (flops_per_step * N_STEPS) / total_compute_s / 1e9

    print(f"\n  FLOPs per forward pass:  {format_flops(total_flops)}")
    print(f"  FLOPs per step (fwd+bwd ≈ 3x): {format_flops(flops_per_step)}")
    print(f"  Achieved throughput:     {gflops_per_sec:.2f} GFLOPS/sec")
    print(f"  (NumPy on CPU; no GPU — peak CPUs achieve ~100-500 GFLOPS)")

    # -----------------------------------------------------------------------
    # 4. Forward vs backward comparison
    # -----------------------------------------------------------------------
    fwd_total_ms  = prof.total_time('forward')  * 1000
    bwd_total_ms  = prof.total_time('backward') * 1000
    ratio = bwd_total_ms / fwd_total_ms if fwd_total_ms > 0 else float('nan')

    print(f"\n  Forward  total: {fwd_total_ms:.1f} ms over {N_STEPS} steps")
    print(f"  Backward total: {bwd_total_ms:.1f} ms over {N_STEPS} steps")
    print(f"  Backward/Forward ratio: {ratio:.2f}x")
    print(f"  (Expected ~2-3x: backward must compute dL/dW for every layer,")
    print(f"   plus store and reuse intermediate activations)")

    # -----------------------------------------------------------------------
    # 5. Memory report
    # -----------------------------------------------------------------------
    print(f"\n  Peak allocated memory (tracemalloc): {peak_mb:.2f} MB")
    breakdown = model_memory_breakdown(LAYER_SIZES, batch_size=BATCH)
    print(f"\n  Theoretical total:  {breakdown['total_mb']:.2f} MB")
    print(f"  (tracemalloc may differ due to Python overhead, NumPy buffers,")
    print(f"   and temporary allocations not captured at peak)")

    # -----------------------------------------------------------------------
    # 6. FLOPs per layer breakdown
    # -----------------------------------------------------------------------
    report_model_flops(LAYER_SIZES, batch_size=BATCH)

    # -----------------------------------------------------------------------
    # 7. Loss curve summary
    # -----------------------------------------------------------------------
    n = len(losses)
    if n > 10:
        print(f"  Loss: step 1={losses[0]:.4f}  step {n//2}={losses[n//2]:.4f}  "
              f"step {n}={losses[-1]:.4f}  (training)")

    print(f"\n  Total wall-clock time: {total_wall:.2f}s")
    print()


if __name__ == "__main__":
    main()
