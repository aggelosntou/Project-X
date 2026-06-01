# Loss Functions From Scratch

All major neural network loss functions implemented in pure NumPy, with exact analytical gradients verified against numerical finite differences.

## Files

| File | Purpose |
|------|---------|
| `src/losses.py` | MSE, MAE, Huber, BCE, CrossEntropy, FocalLoss, TripletLoss |
| `src/verify_gradients.py` | Numerical gradient checking for all losses |
| `src/visualize_losses.py` | Print tables and plots of loss/gradient curves |

## Running

```bash
# Smoke test all losses
python3 src/losses.py

# Run gradient verification (all should PASS)
python3 src/verify_gradients.py

# Visualize loss and gradient curves
python3 src/visualize_losses.py
```

## Loss Functions

| Loss | Use Case | Key Property |
|------|----------|-------------|
| MSE | Regression | Gaussian noise assumption, sensitive to outliers |
| MAE | Robust regression | Laplacian noise, resistant to outliers |
| Huber | Robust regression | Hybrid MSE+MAE, δ controls transition |
| BinaryCE | Binary classification | y ∈ {0,1}, apply sigmoid first |
| CrossEntropy | Multi-class | Combined log-softmax + NLL, numerically stable |
| FocalLoss | Class-imbalanced detection | Down-weights easy examples (γ>0) |
| TripletLoss | Metric learning | Embedding space distance ordering |

## Requirements

```
numpy
```

Optional (for plots): `matplotlib`
