"""
distillation.py — Knowledge distillation: train a small student to mimic a large teacher.

The distillation loss combines:
  - Hard loss: cross-entropy with ground-truth labels (standard training)
  - Soft loss: KL divergence between teacher and student softened distributions

    L_total = alpha * L_hard + (1 - alpha) * T^2 * L_soft

The temperature T "softens" the distributions: high T makes the softmax output
more uniform, revealing inter-class similarities that hard labels hide.
"""

import numpy as np
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
# Softmax utilities
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Numerically stable softmax with temperature scaling.
    Higher T → more uniform distribution (softer).
    T → 0: argmax (one-hot).  T → ∞: uniform.
    """
    x_scaled = x / temperature
    e = np.exp(x_scaled - x_scaled.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Cross-entropy: H(y_true, p) = -sum y_true * log(p)"""
    n = labels.shape[0]
    return -float(np.mean(np.log(probs[np.arange(n), labels] + 1e-9)))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) = sum p * log(p/q)  (student q learns to match teacher p)"""
    return float(np.mean(np.sum(p * np.log(p / (q + 1e-9) + 1e-9), axis=-1)))


# ---------------------------------------------------------------------------
# Distillation Loss
# ---------------------------------------------------------------------------

class DistillationLoss:
    """
    Combined distillation loss as in Hinton et al. (2015).

    L_total = alpha * L_hard + (1 - alpha) * T^2 * L_soft

    The T^2 factor compensates for the fact that soft targets have gradients
    scaled by 1/T^2 compared to hard targets (see paper for derivation).

    Parameters
    ----------
    alpha       : weight on hard loss (1.0 = only hard labels, 0.0 = only soft)
    temperature : T > 1 softens the teacher output, revealing dark knowledge
    """

    def __init__(self, alpha: float = 0.5, temperature: float = 4.0):
        assert 0 <= alpha <= 1, "alpha must be in [0, 1]"
        assert temperature > 0, "temperature must be positive"
        self.alpha       = alpha
        self.temperature = temperature

    def compute(self,
                student_logits: np.ndarray,
                teacher_logits: np.ndarray,
                labels: np.ndarray
                ) -> Tuple[float, float, float]:
        """
        Compute the distillation loss.

        Parameters
        ----------
        student_logits : shape (N, n_classes) — student's raw output
        teacher_logits : shape (N, n_classes) — teacher's raw output (no grad)
        labels         : shape (N,) int64 — ground truth labels

        Returns
        -------
        total_loss : float — weighted combination
        hard_loss  : float — cross-entropy with ground truth
        soft_loss  : float — KL divergence with teacher soft targets
        """
        # Hard loss: cross-entropy with one-hot labels
        student_probs_1 = softmax(student_logits, temperature=1.0)
        hard_loss       = cross_entropy(student_probs_1, labels)

        # Soft loss: KL(teacher_soft || student_soft) at temperature T
        teacher_soft = softmax(teacher_logits, temperature=self.temperature)
        student_soft = softmax(student_logits, temperature=self.temperature)
        soft_loss    = kl_divergence(teacher_soft, student_soft)

        # Combined: T^2 scaling compensates for gradient magnitude change
        total_loss = (self.alpha * hard_loss +
                      (1 - self.alpha) * (self.temperature ** 2) * soft_loss)

        return total_loss, hard_loss, soft_loss

    def gradient(self,
                 student_logits: np.ndarray,
                 teacher_logits: np.ndarray,
                 labels: np.ndarray) -> np.ndarray:
        """
        Compute gradient of the combined loss w.r.t. student_logits.

        The combined gradient is:
            dL/dz_student = alpha * grad_hard + (1-alpha) * T * grad_soft

        Hard gradient:   (student_probs - one_hot) / N
        Soft gradient:   T * (student_soft - teacher_soft) / N
        """
        n = labels.shape[0]

        # Hard gradient
        student_probs = softmax(student_logits, 1.0)
        one_hot       = np.zeros_like(student_probs)
        one_hot[np.arange(n), labels] = 1.0
        grad_hard     = (student_probs - one_hot) / n

        # Soft gradient
        teacher_soft  = softmax(teacher_logits, self.temperature)
        student_soft  = softmax(student_logits, self.temperature)
        # dKL/dz = (1/T) * (student_soft - teacher_soft)
        # Combined: T^2 * (1/T) = T
        grad_soft     = (student_soft - teacher_soft) / n

        return (self.alpha * grad_hard +
                (1 - self.alpha) * self.temperature * grad_soft)


# ---------------------------------------------------------------------------
# MLP for distillation experiments
# ---------------------------------------------------------------------------

class MLP:
    """Simple MLP with manual forward/backward."""

    def __init__(self, layer_sizes: list, rng: np.random.Generator):
        self.weights, self.biases = [], []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(6.0 / (n_in + n_out))
            self.weights.append(rng.uniform(-limit, limit, (n_out, n_in)).astype(np.float64))
            self.biases.append(np.zeros(n_out, dtype=np.float64))

    def logits(self, x: np.ndarray) -> np.ndarray:
        h = x.astype(np.float64)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W.T + b
            h = np.maximum(z, 0) if i < len(self.weights) - 1 else z
        return h

    def predict_class(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(softmax(self.logits(x)), axis=1)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict_class(x) == y))

    def n_params(self) -> int:
        return sum(W.size + b.size for W, b in zip(self.weights, self.biases))


# ---------------------------------------------------------------------------
# Training with distillation
# ---------------------------------------------------------------------------

def train_with_distillation(
    teacher: MLP,
    student: MLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 30,
    lr: float = 0.01,
    batch_size: int = 64,
    alpha: float = 0.5,
    temperature: float = 4.0,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> List[dict]:
    """
    Train the student to mimic the teacher using distillation loss.

    The teacher's weights are frozen (no gradient).
    The student learns from both the ground-truth labels AND the teacher's
    soft probability distributions.

    Returns list of per-epoch metric dicts.
    """
    rng  = rng or np.random.default_rng(0)
    loss_fn = DistillationLoss(alpha=alpha, temperature=temperature)
    n    = x_train.shape[0]
    history = []

    for epoch in range(epochs):
        perm       = rng.permutation(n)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb  = x_train[idx].astype(np.float64)
            yb  = y_train[idx]
            nb  = len(idx)

            # Forward: student
            activations = [xb]
            h = xb
            for i, (W, b) in enumerate(zip(student.weights, student.biases)):
                z = h @ W.T + b
                h = np.maximum(z, 0) if i < len(student.weights) - 1 else z
                activations.append(h)
            student_logits = activations[-1]

            # Teacher forward (no gradient)
            teacher_logits = teacher.logits(xb)

            # Loss + gradient
            total_loss, hard_loss, soft_loss = loss_fn.compute(
                student_logits, teacher_logits, yb
            )
            delta = loss_fn.gradient(student_logits, teacher_logits, yb)

            epoch_loss += total_loss
            n_batches  += 1

            # Backward through student
            for i in reversed(range(len(student.weights))):
                h_in = activations[i]
                dW = delta.T @ h_in
                db = delta.sum(axis=0)
                student.weights[i] -= lr * dW
                student.biases[i]  -= lr * db
                if i > 0:
                    delta = delta @ student.weights[i]
                    delta *= (activations[i] > 0)

        train_acc = student.accuracy(x_train, y_train)
        epoch_metrics = {
            "epoch"      : epoch + 1,
            "loss"       : epoch_loss / max(n_batches, 1),
            "train_acc"  : train_acc,
        }
        history.append(epoch_metrics)

        if verbose and (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:3d}: loss={epoch_metrics['loss']:.4f}  "
                  f"train_acc={train_acc:.3f}")

    return history


def train_standard(
    student: MLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 30,
    lr: float = 0.01,
    batch_size: int = 64,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> List[dict]:
    """Train student with standard cross-entropy (no teacher)."""
    rng = rng or np.random.default_rng(0)
    n   = x_train.shape[0]
    history = []

    for epoch in range(epochs):
        perm = rng.permutation(n)
        loss_sum = 0; nb = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = x_train[idx].astype(np.float64), y_train[idx]
            nbatch = len(idx)

            activations = [xb]
            h = xb
            for i, (W, b) in enumerate(zip(student.weights, student.biases)):
                z = h @ W.T + b
                h = np.maximum(z, 0) if i < len(student.weights) - 1 else z
                activations.append(h)
            logits = activations[-1]

            probs   = softmax(logits, 1.0)
            loss    = cross_entropy(probs, yb)
            loss_sum += loss; nb += 1

            delta = probs.copy()
            delta[np.arange(nbatch), yb] -= 1
            delta /= nbatch

            for i in reversed(range(len(student.weights))):
                h_in = activations[i]
                student.weights[i] -= lr * (delta.T @ h_in)
                student.biases[i]  -= lr * delta.sum(axis=0)
                if i > 0:
                    delta = delta @ student.weights[i]
                    delta *= (activations[i] > 0)

        acc = student.accuracy(x_train, y_train)
        history.append({"epoch": epoch+1, "loss": loss_sum/max(nb,1), "train_acc": acc})
        if verbose and (epoch+1) % 5 == 0:
            print(f"    Epoch {epoch+1:3d}: loss={loss_sum/max(nb,1):.4f}  acc={acc:.3f}")

    return history


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N, D, C = 100, 16, 5
    x = rng.standard_normal((N, D)).astype(np.float32)
    y = rng.integers(0, C, N)

    teacher  = MLP([D, 64, 32, C], rng)
    student1 = MLP([D, 16, C], rng)
    student2 = MLP([D, 16, C], rng.default_rng(1))

    loss_fn = DistillationLoss(alpha=0.5, temperature=4.0)
    s_logits = student1.logits(x)
    t_logits = teacher.logits(x)

    total, hard, soft = loss_fn.compute(s_logits, t_logits, y)
    print(f"Distillation loss — total: {total:.4f}  hard: {hard:.4f}  soft: {soft:.4f}")

    grad = loss_fn.gradient(s_logits, t_logits, y)
    print(f"Gradient shape: {grad.shape}  norm: {np.linalg.norm(grad):.4f}")
    print("Self-test passed.")
