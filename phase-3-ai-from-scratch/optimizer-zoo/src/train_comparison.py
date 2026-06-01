"""
train_comparison.py — Train a small MLP on synthetic classification data
with every optimizer and compare convergence speed and final accuracy.

Dataset: two_moons (implemented in NumPy — no sklearn dependency)
  - 2D input, binary classification
  - Non-linearly separable (requires hidden layers)
  - 1000 training samples, 200 test samples

MLP Architecture:
  Linear(2, 32) → ReLU → Linear(32, 16) → ReLU → Linear(16, 2)

Reports:
  - Final accuracy (test set)
  - Steps to reach 95% training accuracy (or N/A)
  - Loss curve for each optimizer
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from optimizers import SGD, SGDMomentum, Nesterov, RMSProp, Adam, AdamW

# ── Dataset: two-moons implemented in NumPy ────────────────────────────────────

def make_moons(n_samples: int = 1000, noise: float = 0.1, seed: int = 42) -> tuple:
    """
    Generate two interleaved half-circles (moon shapes).

    Returns: (X, y) where X: (n_samples, 2), y: (n_samples,) binary labels
    """
    rng = np.random.RandomState(seed)
    n_each = n_samples // 2

    # Moon 1: top half-circle at (0.5, 0.25)
    theta1 = np.linspace(0, np.pi, n_each)
    X1 = np.c_[np.cos(theta1), np.sin(theta1)]

    # Moon 2: bottom half-circle at (0.0, 0.0), shifted
    theta2 = np.linspace(0, np.pi, n_each)
    X2 = np.c_[1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5]

    X = np.vstack([X1, X2]).astype(np.float64)
    y = np.hstack([np.zeros(n_each, dtype=np.int64),
                   np.ones(n_each,  dtype=np.int64)])

    # Add noise
    X += rng.randn(*X.shape) * noise

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ── MLP layers ────────────────────────────────────────────────────────────────

class Linear:
    def __init__(self, in_f: int, out_f: int):
        std = np.sqrt(2.0 / in_f)
        self.W  = np.random.randn(in_f, out_f) * std
        self.b  = np.zeros(out_f)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x = None

    def forward(self, x): self._x = x; return x @ self.W + self.b
    def backward(self, d):
        self.dW += self._x.T @ d
        self.db += d.sum(0)
        return d @ self.W.T
    def parameters(self): return [(self.W, self.dW), (self.b, self.db)]
    def zero_grad(self):
        self.dW[:] = 0; self.db[:] = 0


class ReLU:
    def forward(self, x): self._m = x > 0; return x * self._m
    def backward(self, d): return d * self._m
    def parameters(self): return []
    def zero_grad(self): pass


class MLP:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(ReLU())

    def forward(self, x):
        for l in self.layers: x = l.forward(x)
        return x

    def backward(self, d):
        for l in reversed(self.layers): d = l.backward(d)

    def parameters(self):
        p = []
        for l in self.layers: p.extend(l.parameters())
        return p

    def zero_grad(self):
        for l in self.layers: l.zero_grad()


def cross_entropy(logits, targets):
    """Softmax + NLL. Returns (loss, d_logits)."""
    N = len(targets)
    shifted = logits - logits.max(1, keepdims=True)
    exp_z   = np.exp(shifted)
    probs   = exp_z / exp_z.sum(1, keepdims=True)
    loss    = -np.log(probs[np.arange(N), targets] + 1e-9).mean()
    d       = probs.copy()
    d[np.arange(N), targets] -= 1
    d /= N
    return loss, d


def accuracy(logits, labels):
    return (logits.argmax(1) == labels).mean()


# ── Training ──────────────────────────────────────────────────────────────────

def train_with_optimizer(optimizer_fn, X_train, y_train, X_test, y_test,
                         n_epochs=500, batch_size=64, seed=42):
    """
    Train MLP with a given optimizer.
    optimizer_fn: callable that returns a fresh optimizer instance.
    """
    np.random.seed(seed)
    model = MLP([2, 32, 16, 2])
    opt   = optimizer_fn()
    N     = len(y_train)

    loss_history  = []
    acc_history   = []
    steps_to_95   = None
    total_steps   = 0

    for epoch in range(n_epochs):
        perm = np.random.permutation(N)
        X_s  = X_train[perm]
        y_s  = y_train[perm]

        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, N, batch_size):
            end    = min(start + batch_size, N)
            x_b    = X_s[start:end]
            y_b    = y_s[start:end]

            logits       = model.forward(x_b)
            loss, d_out  = cross_entropy(logits, y_b)
            model.backward(d_out)

            opt.step(model.parameters())
            model.zero_grad()

            epoch_loss += float(loss)
            n_batches  += 1
            total_steps += 1

        avg_loss   = epoch_loss / n_batches
        train_acc  = accuracy(model.forward(X_train), y_train)

        loss_history.append(avg_loss)
        acc_history.append(float(train_acc))

        if steps_to_95 is None and train_acc >= 0.95:
            steps_to_95 = total_steps

    test_logits = model.forward(X_test)
    test_acc    = float(accuracy(test_logits, y_test))

    return {
        'loss_history': loss_history,
        'acc_history':  acc_history,
        'test_acc':     test_acc,
        'steps_to_95':  steps_to_95,
        'final_train_acc': acc_history[-1] if acc_history else 0.0,
    }


def main():
    print("=" * 65)
    print("OPTIMIZER COMPARISON — Two Moons Classification")
    print("=" * 65)

    # Generate dataset
    X, y = make_moons(n_samples=1200, noise=0.15, seed=42)
    X_train, y_train = X[:1000], y[:1000]
    X_test,  y_test  = X[1000:], y[1000:]

    # Standardize
    mu  = X_train.mean(0, keepdims=True)
    sig = X_train.std(0,  keepdims=True) + 1e-7
    X_train = (X_train - mu) / sig
    X_test  = (X_test  - mu) / sig

    print(f"Train: {X_train.shape},  Test: {X_test.shape}")
    print()

    N_EPOCHS   = 500
    BATCH_SIZE = 64

    optimizer_configs = [
        ("SGD",          lambda: SGD(lr=0.05)),
        ("SGD+Momentum", lambda: SGDMomentum(lr=0.05, momentum=0.9)),
        ("Nesterov",     lambda: Nesterov(lr=0.05, momentum=0.9)),
        ("RMSProp",      lambda: RMSProp(lr=0.01)),
        ("Adam",         lambda: Adam(lr=0.01)),
        ("AdamW",        lambda: AdamW(lr=0.01, weight_decay=0.001)),
    ]

    results = {}
    print(f"Training MLP [2→32→16→2] for {N_EPOCHS} epochs, batch={BATCH_SIZE}\n")

    for name, opt_fn in optimizer_configs:
        print(f"  Training with {name}...", end='', flush=True)
        res = train_with_optimizer(
            opt_fn, X_train, y_train, X_test, y_test,
            n_epochs=N_EPOCHS, batch_size=BATCH_SIZE
        )
        results[name] = res
        print(f" done  (test_acc={res['test_acc']:.2%})")

    # ── Print comparison table ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"  {'Optimizer':<15} | {'Final Train':>11} | {'Test Acc':>8} | {'Steps→95%':>10} | {'Final Loss':>10}")
    print(f"  {'-'*15}-+-{'-'*11}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")

    for name, _ in optimizer_configs:
        r = results[name]
        steps_str = str(r['steps_to_95']) if r['steps_to_95'] else "N/A"
        final_loss = r['loss_history'][-1] if r['loss_history'] else float('nan')
        print(f"  {name:<15} | {r['final_train_acc']:>10.2%} | "
              f"{r['test_acc']:>7.2%} | {steps_str:>10} | {final_loss:>10.4f}")

    # ── Winner ────────────────────────────────────────────────────────────
    best_name = max(results, key=lambda n: results[n]['test_acc'])
    fastest   = min(
        [(n, r['steps_to_95']) for n, r in results.items() if r['steps_to_95'] is not None],
        key=lambda t: t[1],
        default=(None, None)
    )
    print(f"\n  Best test accuracy:    {best_name} ({results[best_name]['test_acc']:.2%})")
    if fastest[0]:
        print(f"  Fastest to 95% train:  {fastest[0]} ({fastest[1]} steps)")

    # ── Plot if matplotlib available ──────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curves
        ax = axes[0]
        for name, _ in optimizer_configs:
            ax.plot(results[name]['loss_history'], label=name, linewidth=1.5)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Training Loss')
        ax.set_title('Loss Curves — Two Moons')
        ax.legend(fontsize=9); ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Accuracy curves
        ax = axes[1]
        for name, _ in optimizer_configs:
            ax.plot(results[name]['acc_history'], label=name, linewidth=1.5)
        ax.axhline(0.95, color='black', linestyle='--', linewidth=1, label='95% target')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Training Accuracy')
        ax.set_title('Accuracy Curves — Two Moons')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(os.path.dirname(__file__), '..', 'optimizer_comparison.png')
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        print(f"\nSaved: {out_path}")
        plt.close()

    except ImportError:
        print("\nmatplotlib not available — skipping plots")

    return results


if __name__ == "__main__":
    np.random.seed(42)
    main()
