"""
train_copy.py — Train a Transformer on the copy task.

TASK
  Input:  [a, b, c, d, e, f, g, h, i, j]   (sequence of 10 random tokens)
  Output: [a, b, c, d, e, f, g, h, i, j]   (exact same sequence)

WHY THIS TASK?
  The copy task is the classic sanity-check for attention mechanisms.
  To solve it, every output position must attend directly to the
  corresponding input position — a perfect test of whether self-attention
  is learning to route information correctly.

  An MLP with no attention would have to memorize each position mapping
  separately (10x harder). A Transformer with attention should solve this
  easily because it can learn "attend to position i to produce token at position i."

TRAINING DETAILS
  Vocabulary:  tokens 0..9 (VOCAB_SIZE = 10), treated as class indices
  Sequence length: SEQ_LEN = 10
  Model:  d_model=32, num_heads=2, d_ff=64, n_blocks=2
  Loss:   cross-entropy at every position
  Optimizer: SGD with manual backprop
  Expected: ~99% accuracy by epoch 300
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from transformer import Transformer, cross_entropy_loss

# ── Hyperparameters ────────────────────────────────────────────────────────────
VOCAB_SIZE  = 10
SEQ_LEN     = 10
BATCH_SIZE  = 32
N_EPOCHS    = 500
LR          = 3e-3
D_MODEL     = 32
NUM_HEADS   = 2
D_FF        = 64
N_BLOCKS    = 2
MAX_LEN     = 20
SEED        = 42

np.random.seed(SEED)


def generate_batch(batch_size: int, seq_len: int, vocab_size: int):
    """
    Generate a batch for the copy task.
    Input and target are the same random integer sequence.
    """
    sequences = np.random.randint(0, vocab_size, (batch_size, seq_len))
    return sequences, sequences  # input == target


def compute_accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Token-level accuracy: fraction of positions where argmax(logits) == target.
    """
    predictions = logits.argmax(axis=-1)  # (batch, seq)
    correct = (predictions == targets).sum()
    total   = targets.size
    return correct / total


def train():
    print("=" * 60)
    print("COPY TASK — Transformer Training")
    print("=" * 60)
    print(f"Vocabulary: {VOCAB_SIZE} tokens | Sequence length: {SEQ_LEN}")
    print(f"Model: d_model={D_MODEL}, heads={NUM_HEADS}, d_ff={D_FF}, blocks={N_BLOCKS}")
    print(f"Batch: {BATCH_SIZE} | Epochs: {N_EPOCHS} | LR: {LR}")
    print()

    model = Transformer(
        vocab_size=VOCAB_SIZE, max_len=MAX_LEN,
        n_blocks=N_BLOCKS, d_model=D_MODEL,
        num_heads=NUM_HEADS, d_ff=D_FF
    )

    n_params = sum(p.size for p, g in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print()

    best_acc = 0.0
    loss_history = []
    acc_history  = []

    for epoch in range(N_EPOCHS):
        inputs, targets = generate_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)

        # Forward
        logits = model.forward(inputs)

        # Loss + gradient
        loss, d_logits = cross_entropy_loss(logits, targets)

        # Backward
        model.backward(d_logits)

        # SGD update (with gradient clipping for stability)
        # Clip by global norm
        all_grads = np.concatenate([g.ravel() for p, g in model.parameters()])
        global_norm = np.sqrt((all_grads ** 2).sum())
        clip_threshold = 1.0
        if global_norm > clip_threshold:
            scale = clip_threshold / (global_norm + 1e-6)
            for p, g in model.parameters():
                g *= scale

        model.sgd_step(LR)

        acc = compute_accuracy(logits, targets)
        loss_history.append(float(loss))
        acc_history.append(float(acc))

        if best_acc < acc:
            best_acc = acc

        if epoch % 50 == 0 or epoch == N_EPOCHS - 1:
            print(f"epoch {epoch:4d} | loss {loss:.4f} | acc {acc:.2%}")

    print()
    print(f"Best accuracy: {best_acc:.2%}")

    # ── Qualitative inspection ────────────────────────────────────────────────
    print("\n=== Sample predictions ===")
    test_inputs, _ = generate_batch(5, SEQ_LEN, VOCAB_SIZE)
    test_logits = model.forward(test_inputs)
    test_preds  = test_logits.argmax(axis=-1)

    for i in range(5):
        inp  = test_inputs[i].tolist()
        pred = test_preds[i].tolist()
        ok   = "✓" if inp == pred else "✗"
        print(f"  Input:  {inp}")
        print(f"  Output: {pred}  {ok}")
        print()

    # ── Final accuracy over larger test set ──────────────────────────────────
    test_inputs, test_targets = generate_batch(1000, SEQ_LEN, VOCAB_SIZE)
    test_logits = model.forward(test_inputs)
    final_acc   = compute_accuracy(test_logits, test_targets)
    print(f"Final test accuracy (N=1000): {final_acc:.2%}")

    if final_acc >= 0.95:
        print("SUCCESS: Model solved the copy task (>= 95% accuracy).")
    else:
        print("Note: Accuracy below 95% — try more epochs or tune LR.")

    return model, loss_history, acc_history


if __name__ == "__main__":
    train()
