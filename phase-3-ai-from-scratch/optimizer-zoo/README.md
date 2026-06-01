# Optimizer Zoo

All major gradient-based optimizers implemented from scratch in NumPy, with comparison on standard benchmark functions and a classification task.

## Files

| File | Purpose |
|------|---------|
| `src/optimizers.py` | SGD, SGDMomentum, Nesterov, RMSProp, Adam, AdamW |
| `src/test_functions.py` | Rosenbrock, Beale, Himmelblau, Rastrigin with exact gradients |
| `src/compare_optimizers.py` | Run all optimizers on all test functions, print table + plot |
| `src/train_comparison.py` | Train MLP on two-moons dataset, compare convergence |

## Running

```bash
# Test optimizer implementations
python3 src/optimizers.py

# Verify gradient implementations
python3 src/test_functions.py

# Compare optimizers on test functions
python3 src/compare_optimizers.py

# Compare on neural network training task
python3 src/train_comparison.py
```

## Optimizers

| Optimizer | Key Feature | Typical lr |
|-----------|------------|-----------|
| SGD | Baseline, no memory | 0.01-0.1 |
| SGD+Momentum | Velocity accumulation | 0.01-0.1 |
| Nesterov | Look-ahead gradient | 0.01-0.1 |
| RMSProp | Adaptive per-parameter lr | 0.001-0.01 |
| Adam | Momentum + adaptive lr | 0.001 |
| AdamW | Adam + decoupled weight decay | 0.001 |

## Requirements

```
numpy
```

Optional (for plots): `matplotlib`
