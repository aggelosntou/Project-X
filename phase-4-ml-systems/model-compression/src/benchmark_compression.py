"""
benchmark_compression.py — Full compression pipeline benchmark.

Trains a teacher MLP, then:
  1. Applies magnitude pruning at 30/50/70/90% sparsity (with fine-tuning)
  2. Trains a small student WITHOUT distillation (baseline)
  3. Trains a small student WITH distillation (using distillation.py)

Prints a comprehensive comparison table.
"""

import copy
import numpy as np
import sys
import os
import time

# Make sure we can import the sibling modules
sys.path.insert(0, os.path.dirname(__file__))

from pruning import magnitude_prune_model, measure_model_sparsity, apply_masks
from distillation import (
    MLP as DistillMLP,
    DistillationLoss,
    softmax,
    cross_entropy,
    train_with_distillation,
    train_standard,
)


# ---------------------------------------------------------------------------
# Self-contained MLP (mirrors distillation.MLP but with explicit train/eval)
# ---------------------------------------------------------------------------

class MLP:
    """
    Simple 3-layer MLP for benchmarking compression.

    Xavier-uniform initialisation; ReLU activations on hidden layers;
    raw logits on the output layer (no final activation).
    """

    def __init__(self, layer_sizes, rng=None):
        """
        Parameters
        ----------
        layer_sizes : list[int]  e.g. [784, 256, 128, 10]
        rng         : np.random.Generator (optional)
        """
        if rng is None:
            rng = np.random.default_rng(42)
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases  = []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Xavier uniform
            limit = np.sqrt(6.0 / (n_in + n_out))
            W = rng.uniform(-limit, limit, (n_out, n_in)).astype(np.float32)
            b = np.zeros(n_out, dtype=np.float32)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : (N, input_dim) float32

        Returns
        -------
        logits : (N, n_classes) float32
        """
        h = x.astype(np.float32)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W.T + b
            # ReLU on all layers except the last
            h = np.maximum(z, 0.0) if i < len(self.weights) - 1 else z
        return h

    def parameters(self):
        """Return list of weight arrays only (not biases) — used for pruning."""
        return self.weights

    def n_params(self):
        return sum(W.size + b.size for W, b in zip(self.weights, self.biases))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def generate_data(n=1000, input_dim=20, n_classes=5, seed=0):
    """
    Generate a synthetic linearly-separable-ish classification dataset.

    Each class has its own mean vector so the problem is non-trivial
    but achievable with a small MLP.
    """
    rng = np.random.default_rng(seed)
    class_means = rng.standard_normal((n_classes, input_dim)) * 2.0
    X_list, y_list = [], []
    per_class = n // n_classes
    for c in range(n_classes):
        X_list.append(rng.standard_normal((per_class, input_dim)) + class_means[c])
        y_list.append(np.full(per_class, c, dtype=np.int64))
    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list)
    # shuffle
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def train(model, X, y, epochs=200, lr=0.01, batch_size=64,
          masks=None, verbose=False, rng=None):
    """
    Simple SGD training loop with cross-entropy loss.

    Parameters
    ----------
    model   : MLP instance (weights modified in place)
    X, y    : training data
    epochs  : number of full passes
    lr      : learning rate
    batch_size
    masks   : list[np.ndarray] — if provided, re-apply after each update
              (maintains pruned-weight structure during fine-tuning)
    verbose : print loss every 50 epochs
    rng     : np.random.Generator

    Returns
    -------
    final accuracy (float)
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(X)

    for epoch in range(epochs):
        perm = rng.permutation(n)

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb  = X[idx].astype(np.float32)
            yb  = y[idx]
            nb  = len(idx)

            # --- Forward ---
            activations = [xb]
            h = xb
            for i, (W, b) in enumerate(zip(model.weights, model.biases)):
                z = h @ W.T + b
                h = np.maximum(z, 0.0) if i < len(model.weights) - 1 else z
                activations.append(h)
            logits = activations[-1]

            # --- Loss: cross-entropy ---
            probs = _softmax(logits.astype(np.float64))
            # gradient of CE w.r.t. logits = (probs - one_hot) / N
            delta = probs.copy()
            delta[np.arange(nb), yb] -= 1.0
            delta = (delta / nb).astype(np.float32)

            # --- Backward ---
            for i in reversed(range(len(model.weights))):
                h_in = activations[i]
                dW = delta.T @ h_in
                db = delta.sum(axis=0)
                model.weights[i] -= lr * dW
                model.biases[i]  -= lr * db
                if i > 0:
                    delta = delta @ model.weights[i]
                    delta = delta * (activations[i] > 0).astype(np.float32)

            # Re-apply masks if provided (lottery-ticket / fine-tune style)
            if masks is not None:
                for i, mask in enumerate(masks):
                    model.weights[i] *= mask.astype(np.float32)

        if verbose and (epoch + 1) % 50 == 0:
            acc = evaluate(model, X, y)
            print(f"    Epoch {epoch+1}: acc={acc:.3f}")

    return evaluate(model, X, y)


def evaluate(model, X, y):
    """Return classification accuracy on (X, y)."""
    logits = model.forward(X)
    preds  = np.argmax(logits, axis=1)
    return float(np.mean(preds == y))


# ---------------------------------------------------------------------------
# Size utilities
# ---------------------------------------------------------------------------

def model_size_bytes(model):
    """
    Total bytes used by all parameters (weights + biases).

    float32 = 4 bytes per element.
    """
    total = 0
    for W, b in zip(model.weights, model.biases):
        total += W.size * W.itemsize
        total += b.size * b.itemsize
    return total


def count_params(model):
    return sum(W.size + b.size for W, b in zip(model.weights, model.biases))


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    np.random.seed(0)
    rng = np.random.default_rng(42)

    # -----------------------------------------------------------------------
    # Use larger synthetic data so results are more stable
    # -----------------------------------------------------------------------
    INPUT_DIM  = 64
    N_CLASSES  = 10
    N_TRAIN    = 2000
    N_TEST     = 500

    print("Generating synthetic data ...")
    X_train, y_train = generate_data(N_TRAIN, INPUT_DIM, N_CLASSES, seed=1)
    X_test,  y_test  = generate_data(N_TEST,  INPUT_DIM, N_CLASSES, seed=2)

    # -----------------------------------------------------------------------
    # 1. Train TEACHER
    # -----------------------------------------------------------------------
    print("\n[1/3] Training teacher (784→256→128→10 scaled to fit data) ...")
    # We use INPUT_DIM instead of 784 for speed, but keep 256/128 hidden dims
    teacher = MLP([INPUT_DIM, 256, 128, N_CLASSES], rng=rng)
    t0 = time.time()
    teacher_acc = train(teacher, X_train, y_train,
                        epochs=300, lr=0.005, batch_size=64,
                        rng=np.random.default_rng(10))
    teacher_train_time = time.time() - t0
    teacher_test_acc = evaluate(teacher, X_test, y_test)
    print(f"  Teacher train acc: {teacher_acc:.3f}  test acc: {teacher_test_acc:.3f}  "
          f"({teacher_train_time:.1f}s)")

    # -----------------------------------------------------------------------
    # 2. Pruning experiments
    # -----------------------------------------------------------------------
    print("\n[2/3] Pruning teacher at various sparsity levels ...")
    sparsity_levels = [0.30, 0.50, 0.70, 0.90]
    pruning_results = []

    for sparsity in sparsity_levels:
        # Start from a fresh copy of teacher weights
        pruned_model = copy.deepcopy(teacher)

        # Apply magnitude pruning to weight matrices only
        pruned_weights, masks = magnitude_prune_model(pruned_model.weights, sparsity)
        pruned_model.weights = pruned_weights

        # Fine-tune for 50 epochs while holding masks fixed
        ft_rng = np.random.default_rng(int(sparsity * 100))
        _ = train(pruned_model, X_train, y_train,
                  epochs=50, lr=0.003, batch_size=64,
                  masks=masks, rng=ft_rng)

        # Measure true sparsity after fine-tuning
        sparsity_stats = measure_model_sparsity(pruned_model.weights)
        actual_sparsity = sparsity_stats["global_sparsity"]
        test_acc = evaluate(pruned_model, X_test, y_test)

        pruning_results.append({
            "label"    : f"Pruned {int(sparsity*100)}%",
            "sparsity" : actual_sparsity,
            "test_acc" : test_acc,
            "n_params" : count_params(pruned_model),
            "size_kb"  : model_size_bytes(pruned_model) / 1024,
        })
        print(f"  sparsity={sparsity:.0%}  actual={actual_sparsity:.1%}  "
              f"test_acc={test_acc:.3f}")

    # -----------------------------------------------------------------------
    # 3. Distillation experiments
    # -----------------------------------------------------------------------
    print("\n[3/3] Distillation: training student models ...")
    STUDENT_ARCH = [INPUT_DIM, 64, 32, N_CLASSES]

    # --- Wrap teacher as distillation.MLP for API compatibility ---
    # distillation.MLP.logits() does the same forward pass
    # We'll build a DistillMLP with the same weights
    dist_teacher = DistillMLP(teacher.layer_sizes, rng=np.random.default_rng(99))
    for i in range(len(teacher.weights)):
        dist_teacher.weights[i] = teacher.weights[i].astype(np.float64)
        dist_teacher.biases[i]  = teacher.biases[i].astype(np.float64)

    # 3a. Student trained WITHOUT distillation (plain cross-entropy)
    print("  Training student WITHOUT distillation ...")
    student_no_kd = DistillMLP(STUDENT_ARCH, rng=np.random.default_rng(7))
    train_standard(student_no_kd, X_train, y_train,
                   epochs=200, lr=0.01, batch_size=64,
                   rng=np.random.default_rng(7), verbose=False)
    no_kd_acc = student_no_kd.accuracy(X_test, y_test)
    print(f"    Student (no KD) test acc: {no_kd_acc:.3f}")

    # 3b. Student trained WITH distillation
    print("  Training student WITH distillation (T=4, alpha=0.5) ...")
    student_kd = DistillMLP(STUDENT_ARCH, rng=np.random.default_rng(8))
    train_with_distillation(dist_teacher, student_kd,
                            X_train, y_train,
                            epochs=200, lr=0.01, batch_size=64,
                            alpha=0.5, temperature=4.0,
                            rng=np.random.default_rng(8), verbose=False)
    kd_acc = student_kd.accuracy(X_test, y_test)
    print(f"    Student (KD)    test acc: {kd_acc:.3f}")

    # -----------------------------------------------------------------------
    # Build student parameter count (DistillMLP stores weights + biases)
    # -----------------------------------------------------------------------
    def dist_mlp_size_kb(m):
        total = sum(W.size * 8 + b.size * 8   # float64
                    for W, b in zip(m.weights, m.biases))
        return total / 1024

    def dist_mlp_params(m):
        return sum(W.size + b.size for W, b in zip(m.weights, m.biases))

    # -----------------------------------------------------------------------
    # Print comprehensive results table
    # -----------------------------------------------------------------------
    print()
    print("=" * 75)
    print("  COMPRESSION BENCHMARK RESULTS")
    print("=" * 75)
    fmt_head = f"{'Model':<20} {'Params':>8} {'Size(KB)':>10} {'TestAcc':>9} {'Sparsity':>10}"
    print(fmt_head)
    print("-" * 75)

    def fmt_row(label, params, size_kb, acc, sparsity):
        return (f"{label:<20} {params:>8,} {size_kb:>10.1f} "
                f"{acc:>9.3f} {sparsity:>9.1%}")

    # Teacher
    print(fmt_row("Teacher",
                  count_params(teacher),
                  model_size_bytes(teacher) / 1024,
                  teacher_test_acc,
                  0.0))

    # Pruned models
    for r in pruning_results:
        print(fmt_row(r["label"],
                      r["n_params"],
                      r["size_kb"],
                      r["test_acc"],
                      r["sparsity"]))

    # Students
    print(fmt_row("Student (no KD)",
                  dist_mlp_params(student_no_kd),
                  dist_mlp_size_kb(student_no_kd),
                  no_kd_acc,
                  0.0))
    print(fmt_row("Student (KD)",
                  dist_mlp_params(student_kd),
                  dist_mlp_size_kb(student_kd),
                  kd_acc,
                  0.0))

    print("=" * 75)
    print()

    # -----------------------------------------------------------------------
    # Summary observations
    # -----------------------------------------------------------------------
    kd_gain = kd_acc - no_kd_acc
    prune90 = next(r for r in pruning_results if "90" in r["label"])
    acc_drop_90 = teacher_test_acc - prune90["test_acc"]

    print("Key observations:")
    print(f"  - Pruning at 90% sparsity costs {acc_drop_90:.1%} accuracy vs teacher")
    print(f"  - KD student gains {kd_gain:+.1%} accuracy vs plain student")
    print(f"  - Student is {count_params(teacher) // dist_mlp_params(student_kd)}x "
          f"smaller in parameter count than teacher")
    print()
    print("Note: pruning does NOT reduce wall-clock inference time on standard")
    print("  CPUs/GPUs until sparsity >95% because dense GEMM routines ignore zeros.")
    print("  Real speedups require sparse tensor kernels (e.g. NVIDIA cuSPARSE).")


if __name__ == "__main__":
    main()
