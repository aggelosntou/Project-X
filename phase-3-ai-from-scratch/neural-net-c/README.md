# Neural Network in C

A complete backpropagation neural network in pure C11, no external libraries.

## Build & Run

```bash
make train_xor           # 2-4-1 network solves XOR
make train_mnist_lite    # 784-128-64-10 synthetic classification
```

## Architecture

- Xavier weight initialization for stable training
- Configurable hidden activations: ReLU, tanh, sigmoid, linear
- Full backpropagation with stored pre-activations
- SGD weight update

## Mathematical Foundation

Forward pass for layer l:
```
z_l = a_{l-1} @ W_l + b_l      (linear transformation)
a_l = f(z_l)                    (element-wise activation)
```

Backward pass (chain rule in matrix form):
```
delta_L = (2/N)(a_L - target)                          (MSE gradient at output)
delta_l = (delta_{l+1} @ W_{l+1}^T) ⊙ f'(z_l)       (propagate + gate)
dL/dW_l = a_{l-1}^T @ delta_l                         (weight gradient)
dL/db_l = mean(delta_l, axis=0)                        (bias gradient)
```

SGD update:
```
W_l = W_l - lr * dL/dW_l
```
