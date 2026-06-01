"""
train_addition.py — Train a Transformer to perform integer addition.

TASK
  Given a string like "123+456=" the model must predict "579".
  This tests whether the Transformer can:
    1. Learn digit-by-digit arithmetic (carry propagation)
    2. Generalize to unseen number combinations
    3. Use positional encoding to align input/output digits

FORMAT
  Full string: "123+456=579"
  Input to model: full string (teacher-forced)
  Prediction target: shift-by-one (predict next character at each position)
  Accuracy metric: exact match on the result digits (after '=')

DATASET
  Random pairs (a, b) where 0 <= a, b < 1000
  Result: a + b (up to 4 digits, e.g. 999+999=1998)
  Padded to fixed length MAX_LEN with '_' padding character

WHY IS THIS HARD?
  Carry propagation is inherently right-to-left, but the model reads
  left-to-right. The Transformer must learn to look ahead/back using
  its bidirectional attention. This is a well-known benchmark showing
  that length generalization is limited in standard Transformers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from tokenizer import CharTokenizer
from transformer import Transformer, cross_entropy_loss

# ── Hyperparameters ────────────────────────────────────────────────────────────
MAX_A       = 1000   # a, b drawn from [0, MAX_A)
N_TRAIN     = 5000   # training examples (generated fresh each epoch)
N_TEST      = 500
N_EPOCHS    = 2000
BATCH_SIZE  = 64
LR          = 1e-3
D_MODEL     = 64
NUM_HEADS   = 4
D_FF        = 128
N_BLOCKS    = 3
SEED        = 42

np.random.seed(SEED)


# ── Data generation ────────────────────────────────────────────────────────────

def make_example(a: int, b: int, max_len: int, pad_char: str = '_') -> str:
    """
    Format: "NNN+NNN=NNNN" padded to max_len.
    Example: "  7+999=1006" → "  7+999=1006_"
    """
    s = f"{a}+{b}={a+b}"
    if len(s) < max_len:
        s = s + pad_char * (max_len - len(s))
    return s[:max_len]


def determine_max_len(max_a: int) -> int:
    """
    Max length of any addition string:
    Longest input: "999+999=1998" → 12 chars
    Add 2 for safety.
    """
    max_sum = 2 * (max_a - 1)
    return len(f"{max_a-1}+{max_a-1}={max_sum}") + 2


def build_corpus(n: int, max_a: int, max_len: int) -> list:
    """Generate n random addition examples."""
    examples = []
    for _ in range(n):
        a = np.random.randint(0, max_a)
        b = np.random.randint(0, max_a)
        examples.append(make_example(a, b, max_len))
    return examples


# ── Tokenizer setup ────────────────────────────────────────────────────────────

def build_tokenizer(max_a: int, max_len: int) -> CharTokenizer:
    """Build vocab from all possible characters."""
    # Include all digits, '+', '=', '_'
    alphabet = "0123456789+=" + "_"
    return CharTokenizer(alphabet)


# ── Training helpers ───────────────────────────────────────────────────────────

def examples_to_arrays(examples: list, tok: CharTokenizer, max_len: int):
    """
    Convert list of string examples to (inputs, targets) arrays.

    Language-modelling style: input[t] predicts input[t+1].
    So:
      inputs  = encoded[:-1]   (all tokens except last)
      targets = encoded[1:]    (all tokens except first)
    """
    inputs_list  = []
    targets_list = []
    for ex in examples:
        enc = tok.encode(ex)
        # Ensure length
        if len(enc) < max_len:
            enc = enc + [tok.encode('_')[0]] * (max_len - len(enc))
        enc = enc[:max_len]
        inputs_list.append(enc[:-1])
        targets_list.append(enc[1:])
    return np.array(inputs_list), np.array(targets_list)


def compute_result_accuracy(logits: np.ndarray, targets: np.ndarray,
                             examples: list, tok: CharTokenizer) -> float:
    """
    Exact-match accuracy on the RESULT digits only (after '=').
    Compare the predicted characters after '=' to ground truth.
    """
    preds = logits.argmax(axis=-1)   # (batch, seq-1)
    n_correct = 0
    n_total   = 0

    for i, ex in enumerate(examples):
        eq_pos = ex.index('=')
        # In the shifted (input[:-1] → target[1:]) scheme:
        # '=' is at position eq_pos in the original string,
        # which becomes position eq_pos in targets (predict from token eq_pos onward).
        result_start = eq_pos  # first result digit is predicted at this position in targets
        result_str   = ex[eq_pos+1:].rstrip('_')

        if not result_str:
            continue

        predicted_chars = []
        for pos_offset in range(len(result_str)):
            t = result_start + pos_offset
            if t < preds.shape[1]:
                predicted_chars.append(tok.decode([preds[i, t]]))
            else:
                predicted_chars.append('?')

        predicted_result = ''.join(predicted_chars).rstrip('_')
        if predicted_result == result_str:
            n_correct += 1
        n_total += 1

    return n_correct / max(n_total, 1)


def train():
    MAX_LEN = determine_max_len(MAX_A)
    SEQ_LEN = MAX_LEN - 1   # after shift

    print("=" * 60)
    print("CHARACTER-LEVEL ADDITION — Transformer Training")
    print("=" * 60)
    print(f"Range: 0..{MAX_A-1} + 0..{MAX_A-1}")
    print(f"Max string length: {MAX_LEN}")
    print(f"Model: d_model={D_MODEL}, heads={NUM_HEADS}, d_ff={D_FF}, blocks={N_BLOCKS}")
    print(f"Epochs: {N_EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}")
    print()

    tok = build_tokenizer(MAX_A, MAX_LEN)
    VOCAB_SIZE = tok.vocab_size
    print(f"Vocabulary ({VOCAB_SIZE} chars): {list(tok._char_to_id.keys())}")
    print()

    # Fixed test set
    test_examples = build_corpus(N_TEST, MAX_A, MAX_LEN)
    test_inputs, test_targets = examples_to_arrays(test_examples, tok, MAX_LEN)

    model = Transformer(
        vocab_size=VOCAB_SIZE, max_len=MAX_LEN + 2,
        n_blocks=N_BLOCKS, d_model=D_MODEL,
        num_heads=NUM_HEADS, d_ff=D_FF
    )
    n_params = sum(p.size for p, g in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print()

    # Adam state
    adam_t = 0

    for epoch in range(1, N_EPOCHS + 1):
        # Sample fresh training data each epoch
        train_examples = build_corpus(N_TRAIN, MAX_A, MAX_LEN)
        train_inputs, train_targets = examples_to_arrays(train_examples, tok, MAX_LEN)

        # Mini-batch training
        n_batches = N_TRAIN // BATCH_SIZE
        epoch_loss = 0.0

        idx = np.random.permutation(N_TRAIN)
        train_inputs  = train_inputs[idx]
        train_targets = train_targets[idx]

        for b in range(n_batches):
            x = train_inputs[b * BATCH_SIZE:(b+1) * BATCH_SIZE]
            y = train_targets[b * BATCH_SIZE:(b+1) * BATCH_SIZE]

            logits = model.forward(x)
            loss, d_logits = cross_entropy_loss(logits, y)
            model.backward(d_logits)

            # Gradient clipping
            all_grads = np.concatenate([g.ravel() for p, g in model.parameters()])
            global_norm = np.sqrt((all_grads ** 2).sum())
            if global_norm > 1.0:
                scale = 1.0 / (global_norm + 1e-6)
                for p, g in model.parameters():
                    g *= scale

            adam_t += 1
            model.adam_step(LR, adam_t)
            epoch_loss += float(loss)

        if epoch % 100 == 0 or epoch == 1:
            avg_loss = epoch_loss / n_batches

            # Evaluate on test set (in batches)
            test_logits_list = []
            for b in range(0, N_TEST, BATCH_SIZE):
                xl = test_inputs[b:b+BATCH_SIZE]
                tl = model.forward(xl)
                test_logits_list.append(tl)
            test_logits_all = np.concatenate(test_logits_list, axis=0)

            acc = compute_result_accuracy(
                test_logits_all, test_targets, test_examples, tok
            )

            print(f"epoch {epoch:4d} | loss {avg_loss:.4f} | result_acc {acc:.2%}")

    # ── Final evaluation ────────────────────────────────────────────────────────
    print("\n=== Sample predictions ===")
    demo_examples = [
        make_example(1, 1, MAX_LEN),
        make_example(12, 34, MAX_LEN),
        make_example(100, 200, MAX_LEN),
        make_example(123, 456, MAX_LEN),
        make_example(999, 1, MAX_LEN),
        make_example(500, 500, MAX_LEN),
    ]
    demo_inputs, demo_targets = examples_to_arrays(demo_examples, tok, MAX_LEN)

    for i, ex in enumerate(demo_examples):
        xl = demo_inputs[i:i+1]
        logits = model.forward(xl)
        preds = logits.argmax(axis=-1)[0]
        pred_str = tok.decode(preds)

        # Extract result after '='
        eq_pos = ex.index('=')
        gt_result = ex[eq_pos+1:].rstrip('_')

        # Predicted: decode from position eq_pos
        pred_result = pred_str[eq_pos:eq_pos + len(gt_result)]
        ok = "✓" if pred_result == gt_result else "✗"

        input_str = ex[:eq_pos+1]
        print(f"  {input_str}{gt_result}  →  predicted: '{pred_result}'  {ok}")

    # Final accuracy
    test_logits_list = []
    for b in range(0, N_TEST, BATCH_SIZE):
        xl = test_inputs[b:b+BATCH_SIZE]
        tl = model.forward(xl)
        test_logits_list.append(tl)
    test_logits_all = np.concatenate(test_logits_list, axis=0)
    final_acc = compute_result_accuracy(test_logits_all, test_targets, test_examples, tok)
    print(f"\nFinal test accuracy (exact match, N={N_TEST}): {final_acc:.2%}")


if __name__ == "__main__":
    train()
