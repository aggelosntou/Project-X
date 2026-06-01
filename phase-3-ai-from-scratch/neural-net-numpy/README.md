# Neural Network in NumPy

The same neural network as neural-net-c, written in NumPy. The goal is to see the identical mathematics at a higher level of abstraction.

## Run

```bash
cd src
python3 train_xor.py                # XOR, should reach <0.01 loss
python3 train_mnist.py              # MNIST, should reach >95% test accuracy
python3 compare_with_pytorch.py     # verify against PyTorch (if installed)
```

## Files

- `src/layers.py` — Linear, ReLU, Sigmoid, Tanh, Softmax with forward + backward
- `src/losses.py` — MSELoss, CrossEntropyLoss
- `src/network.py` — Sequential container
- `src/optimizers.py` — SGD, SGDMomentum, Adam
- `src/train_xor.py` — XOR demo
- `src/train_mnist.py` — MNIST training with confusion matrix
- `src/compare_with_pytorch.py` — loss curve comparison

## Key Equations

```
Forward:   z = x @ W + b,   a = f(z)
Backward:  dL/dW = x^T @ delta,   dL/dx = delta @ W^T
```
