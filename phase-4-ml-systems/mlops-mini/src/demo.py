"""
demo.py — End-to-end MLOps mini platform demo.

Runs 5 training experiments with different hyperparameters, tracks them all
in a local SQLite database, registers the best model, and shows the dashboard.

Run:
    python src/demo.py
"""

import numpy as np
import sys
import os
import tempfile
import shutil

# Add src directory to path for sibling imports
sys.path.insert(0, os.path.dirname(__file__))

from tracker        import Experiment, list_experiments, compare_experiments
from model_registry import ModelRegistry
from dashboard      import (print_experiments_table, print_model_registry,
                            print_loss_curve_ascii)


# ---------------------------------------------------------------------------
# Clean up any leftover state from previous demo runs
# ---------------------------------------------------------------------------

DB_PATH    = os.path.join(tempfile.gettempdir(), "mlops_demo.db")
MODEL_DIR  = os.path.join(tempfile.gettempdir(), "mlops_demo_models/")

if os.path.exists(DB_PATH):
    os.unlink(DB_PATH)
if os.path.exists(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)


# ---------------------------------------------------------------------------
# Self-contained MLP (no imports from other subprojects)
# ---------------------------------------------------------------------------

def _softmax(x):
    x = x.astype(np.float64)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


class MLP:
    """Simple 2-hidden-layer MLP for the demo."""

    def __init__(self, in_dim, hidden, out_dim, rng):
        self.W = []
        self.b = []
        sizes = [in_dim, hidden, hidden // 2, out_dim]
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            lim = np.sqrt(6.0 / (n_in + n_out))
            self.W.append(rng.uniform(-lim, lim, (n_out, n_in)).astype(np.float32))
            self.b.append(np.zeros(n_out, dtype=np.float32))

    def forward(self, x):
        self._acts = [x.astype(np.float32)]
        h = self._acts[0]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W.T + b
            h = np.maximum(z, 0.0) if i < len(self.W) - 1 else z
            self._acts.append(h)
        return self._acts[-1]

    def backward_and_update(self, labels, lr):
        n      = len(labels)
        probs  = _softmax(self._acts[-1])
        delta  = probs.copy()
        delta[np.arange(n), labels] -= 1.0
        delta /= n
        delta  = delta.astype(np.float32)

        for i in reversed(range(len(self.W))):
            dW = delta.T @ self._acts[i]
            db = delta.sum(axis=0)
            self.W[i] -= lr * dW
            self.b[i]  -= lr * db
            if i > 0:
                delta = delta @ self.W[i]
                delta = delta * (self._acts[i] > 0).astype(np.float32)

    def accuracy(self, x, y):
        logits = self.forward(x)
        return float(np.mean(np.argmax(logits, axis=1) == y))


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def make_data(n=1500, in_dim=32, n_classes=8, seed=0):
    rng = np.random.default_rng(seed)
    # Class means spread out so the problem is learnable
    means = rng.standard_normal((n_classes, in_dim)) * 3.0
    X_parts, y_parts = [], []
    per_class = n // n_classes
    for c in range(n_classes):
        X_parts.append(rng.standard_normal((per_class, in_dim)) + means[c])
        y_parts.append(np.full(per_class, c, dtype=np.int64))
    X = np.vstack(X_parts).astype(np.float32)
    y = np.concatenate(y_parts)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


print("Generating dataset ...")
X_all, y_all = make_data(n=2000, in_dim=32, n_classes=8, seed=1)
split = int(0.8 * len(X_all))
X_train, y_train = X_all[:split], y_all[:split]
X_val,   y_val   = X_all[split:], y_all[split:]


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_and_track(name: str, lr: float, hidden_size: int,
                    epochs: int = 100, batch_size: int = 64,
                    seed: int = 42):
    """
    Train an MLP, log metrics every 10 epochs to the experiment tracker.

    Returns (run_id, best_val_acc)
    """
    rng = np.random.default_rng(seed)
    model = MLP(in_dim=X_train.shape[1],
                hidden=hidden_size,
                out_dim=int(y_train.max()) + 1,
                rng=rng)

    exp = Experiment(
        name,
        config={'lr': lr, 'hidden': hidden_size, 'epochs': epochs},
        db_path=DB_PATH
    ).start()

    n         = len(X_train)
    best_val  = 0.0

    for epoch in range(epochs):
        # SGD epoch
        perm = rng.permutation(n)
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb  = X_train[idx]
            yb  = y_train[idx]
            nb  = len(yb)

            logits = model.forward(xb)
            probs  = _softmax(logits).astype(np.float64)
            loss   = -float(np.mean(np.log(
                probs[np.arange(nb), yb] + 1e-9)))
            total_loss += loss
            n_batches  += 1

            model.backward_and_update(yb, lr)

        avg_loss = total_loss / max(n_batches, 1)
        val_acc  = model.accuracy(X_val, y_val)
        train_acc = model.accuracy(X_train, y_train)

        if val_acc > best_val:
            best_val = val_acc

        # Log every 10 epochs (and the final epoch)
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            exp.log(epoch + 1, {
                'train_loss': avg_loss,
                'train_acc' : train_acc,
                'val_acc'   : val_acc,
            })

    exp.finish()
    print(f"  {name:<20} lr={lr}  hidden={hidden_size:<4}  "
          f"val_acc={best_val:.4f}")
    return exp.run_id, best_val


# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------

configs = [
    ('run_lr001',  0.001, 64),
    ('run_lr003',  0.003, 64),
    ('run_lr01',   0.010, 64),
    ('run_big',    0.003, 128),
    ('run_small',  0.003, 32),
]

print("\n" + "=" * 55)
print("  Training 5 experiments ...")
print("=" * 55)

run_ids  = []
for i, (name, lr, hidden) in enumerate(configs):
    rid, acc = train_and_track(name, lr, hidden, epochs=80, seed=i)
    run_ids.append((rid, acc, name))

# ---------------------------------------------------------------------------
# Find best run
# ---------------------------------------------------------------------------

best_rid, best_acc, best_name = max(run_ids, key=lambda x: x[1])
print(f"\n  Best run: '{best_name}'  val_acc={best_acc:.4f}")

# ---------------------------------------------------------------------------
# Register best model
# ---------------------------------------------------------------------------

model_path = os.path.join(tempfile.gettempdir(), f"{best_name}_weights.npy")
np.save(model_path, np.zeros(10))   # placeholder weights file

registry = ModelRegistry(db_path=DB_PATH, model_dir=MODEL_DIR)
registry.register(
    'my_mlp', 'v1.0', model_path,
    metrics={'val_acc': float(best_acc)},
    config={'name': best_name, 'source': 'demo.py'}
)
registry.transition_stage('my_mlp', 'v1.0', 'staging')
registry.transition_stage('my_mlp', 'v1.0', 'production')

# Also register a second version to demonstrate stage transitions
model_path2 = os.path.join(tempfile.gettempdir(), "my_mlp_v2_weights.npy")
np.save(model_path2, np.zeros(10))
registry.register(
    'my_mlp', 'v2.0', model_path2,
    metrics={'val_acc': float(best_acc * 0.98)},   # slightly worse
    config={'name': 'experimental', 'source': 'demo.py'}
)
registry.transition_stage('my_mlp', 'v2.0', 'staging')
# v1.0 remains production

# ---------------------------------------------------------------------------
# Show dashboard
# ---------------------------------------------------------------------------

print("\n" + "=" * 55)
print("  === Experiment Dashboard ===")
print("=" * 55)

print_experiments_table(db_path=DB_PATH)
compare_experiments([r for r, _, _ in run_ids], metric='val_acc',
                    db_path=DB_PATH)
print_loss_curve_ascii(best_rid, 'train_loss', db_path=DB_PATH)
print_loss_curve_ascii(best_rid, 'val_acc',    db_path=DB_PATH)

print("\n" + "=" * 55)
print("  === Model Registry ===")
print("=" * 55)

print_model_registry(db_path=DB_PATH, model_dir=MODEL_DIR)
registry.list_versions('my_mlp')

prod = registry.get_production('my_mlp')
print(f"\n  Serving model: {prod['name']} {prod['version']}  "
      f"val_acc={prod['metrics']['val_acc']:.4f}")

# ---------------------------------------------------------------------------
# Cleanup (optional — comment out to inspect the DB afterwards)
# ---------------------------------------------------------------------------

print(f"\n  Demo complete. Database at: {DB_PATH}")
print(f"  Model registry at: {MODEL_DIR}")
print("  (Files will be cleaned up automatically on next run.)")
