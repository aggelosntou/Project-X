# CNN From Scratch

A complete 2D Convolutional Neural Network implemented in pure NumPy using the im2col technique.

## Files

| File | Purpose |
|------|---------|
| `src/conv_layer.py` | Conv2d using im2col for efficient GEMM-based convolution |
| `src/pooling.py` | MaxPool2d and AvgPool2d with full backward pass |
| `src/flatten.py` | Reshape layer bridging conv and linear layers |
| `src/network.py` | Linear, ReLU, Sequential container, Adam optimizer, cross-entropy |
| `src/train_mnist_cnn.py` | Full MNIST training pipeline (target: >95% test accuracy) |

## Running

```bash
# Smoke test individual layers
python3 src/conv_layer.py
python3 src/pooling.py
python3 src/network.py

# Train on MNIST (downloads ~12MB data automatically, ~10-20 min on CPU)
python3 src/train_mnist_cnn.py
```

## Architecture

```
Input (N, 1, 28, 28)
      │
Conv2d(1→8, k=3, pad=1) + ReLU + MaxPool(2)    → (N, 8, 14, 14)
      │
Conv2d(8→16, k=3, pad=1) + ReLU + MaxPool(2)   → (N, 16, 7, 7)
      │
Flatten                                          → (N, 784)
      │
Linear(784→64) + ReLU                           → (N, 64)
      │
Linear(64→10)                                    → (N, 10)  logits
```

**Parameters:** ~52,000 (vs ~200,000+ for an equivalent MLP — due to weight sharing)

## Requirements

```
numpy
```
