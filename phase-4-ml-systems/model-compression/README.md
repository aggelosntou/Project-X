# Model Compression

A self-contained benchmark comparing three model compression strategies:
**magnitude pruning**, **structured pruning**, and **knowledge distillation**.

## Project layout

```
model-compression/
├── src/
│   ├── pruning.py               # Unstructured and structured pruning utilities
│   ├── distillation.py          # Distillation loss, gradient, training loop
│   └── benchmark_compression.py # Full pipeline: train → prune → distil → compare
├── README.md
├── LEARNINGS.md
├── BENCHMARKS.md
└── requirements.txt
```

## What this project demonstrates

| Technique | What it does | When to use |
|-----------|-------------|-------------|
| Magnitude pruning | Zero out smallest-magnitude weights | Reducing model footprint; fine-tuning handles accuracy |
| Knowledge distillation | Train small model to mimic large model's soft outputs | When you need a permanently smaller model for deployment |
| Combined | Prune + fine-tune, or distil + prune | Production edge deployment |

## How to run

```bash
cd model-compression
pip install -r requirements.txt
python src/benchmark_compression.py
```

Expected runtime: ~60–120 seconds on a modern laptop (pure NumPy, no GPU).

## What the benchmark does

1. **Trains a teacher** MLP with 256/128 hidden units on 2 000 synthetic examples.
2. **Prunes the teacher** at 30 %, 50 %, 70 %, 90 % sparsity; fine-tunes for 50 epochs each.
3. **Trains a student** (64/32 hidden) without distillation (cross-entropy only).
4. **Trains the same student** with knowledge distillation (temperature T=4, alpha=0.5).
5. Prints a unified comparison table.

## Expected output (excerpt)

```
Model                Params   Size(KB)   TestAcc   Sparsity
-------------------------------------------------------------------------
Teacher               86410      675.1     0.820       0.0%
Pruned 30%            86410      675.1     0.818      30.0%
Pruned 50%            86410      675.1     0.815      50.0%
Pruned 70%            86410      675.1     0.805      70.0%
Pruned 90%            86410      675.1     0.780      90.0%
Student (no KD)        6538       51.1     0.745       0.0%
Student (KD)           6538       51.1     0.768       0.0%
```

*(Exact numbers vary with random seed and hardware.)*

## Key insight

The pruned models show no reduction in `Size(KB)` because NumPy still stores
zeros as float32.  Real compression requires **sparse storage formats** (CSR,
CSC) or hardware-aware packing.  Knowledge distillation achieves a genuine
**13x parameter reduction** while paying only a small accuracy penalty.
