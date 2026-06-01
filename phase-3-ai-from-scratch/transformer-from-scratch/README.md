# Transformer From Scratch

A complete encoder-only Transformer implemented in pure NumPy, built to understand every component from first principles.

## Files

| File | Purpose |
|------|---------|
| `src/attention.py` | Scaled dot-product attention, MultiHeadAttention |
| `src/layers.py` | LayerNorm, FeedForward, PositionalEncoding, Embedding |
| `src/transformer.py` | EncoderBlock, Encoder, full Transformer + cross-entropy loss |
| `src/tokenizer.py` | CharTokenizer and SimpleTokenizer |
| `src/train_copy.py` | Copy task (sanity-check for attention) |
| `src/train_addition.py` | Character-level integer addition |

## Running

```bash
# Smoke test the model
python3 src/transformer.py

# Train on copy task (should reach >95% accuracy)
python3 src/train_copy.py

# Train on addition (2000 epochs, may take a few minutes)
python3 src/train_addition.py
```

## Architecture

```
Tokens (batch, seq)
      │
      ▼
Embedding (vocab_size → d_model)
      │
      ▼
+ PositionalEncoding (fixed sinusoidal)
      │
      ▼
┌─────────────────────────┐
│  EncoderBlock × N       │
│                         │
│  x = x + Attn(Norm(x)) │  ← Pre-norm residual
│  x = x + FFN(Norm(x))  │  ← Pre-norm residual
└─────────────────────────┘
      │
      ▼
LayerNorm (final)
      │
      ▼
Linear (d_model → vocab_size)
      │
      ▼
Logits (batch, seq, vocab_size)
```

## Requirements

```
numpy
```
