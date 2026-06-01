# Learnings: Autograd Engine

## 1. What is a Computation Graph?

Every arithmetic operation creates a node in a DAG. The DAG records:
- **Nodes**: intermediate values (each a `Value` object)
- **Edges**: which values were combined to create each new value (`_prev` set)
- **Labels**: which operation created each node (`_op`)

Example for `L = (a * b) + c`:
```
a → [*] → [+] → L
b ↗       ↑
          c
```

This graph is built dynamically during the forward pass (eager execution), unlike TensorFlow 1.x which required explicit graph construction.

## 2. Why Topological Sort?

The chain rule requires: to compute `dL/dx`, we must know `dL/dy` for every `y` that `x` feeds into.

Topological order of the graph (parents last) ensures that when we call `node._backward()`, the `node.grad` has already received all contributions from downstream nodes.

If we processed nodes in the wrong order, we'd propagate incomplete (zero) gradients.

## 3. What `_backward` Closures Capture

Each `_backward` is a closure that "closes over" the local variables from the forward pass. For multiplication:

```python
def __mul__(self, other):
    out = Value(self.data * other.data, ...)

    def _backward():
        self.grad  += other.data * out.grad   # captures: self, other, out
        other.grad += self.data  * out.grad
    
    out._backward = _backward
```

The closure captures `self`, `other`, and `out` by reference. When called during backward, `out.grad` is already set (from the downstream node's backward call), and the closure multiplies it by the local partial derivative.

## 4. Chain Rule as Function Composition

For `L = f(g(x))`:
```
dL/dx = (dL/df) * (df/dg) * (dg/dx)
```

In our engine:
- `out.grad` = dL/d(out)  — the gradient flowing in from downstream
- `_backward` computes: `self.grad += (dL/d(out)) * (d(out)/d(self))`

Each `_backward` only implements the LOCAL partial derivative `d(out)/d(input)`. The chain rule is implemented by the sequential calling of all `_backward` functions in reverse topological order — each one multiplies by the local factor and accumulates into `.grad`.

## 5. Gradient Accumulation (`+=` not `=`)

We use `+=` in backward:
```python
self.grad += other.data * out.grad  # NOT self.grad = ...
```

Why? A node can be used multiple times in the graph:
```python
x = Value(3.0)
L = x * x   # x appears twice
```
The gradient `dL/dx = 2x = 6` comes from TWO paths back to `x`. Each path contributes `x = 3`, and `+=` accumulates both: `3 + 3 = 6`.

If we used `=` instead of `+=`, the second contribution would overwrite the first, giving the wrong gradient.

## 6. Scalar Engine vs Tensor Engine

This engine operates on **scalar** Values — one number at a time. PyTorch operates on **tensors** — arrays of numbers.

The mathematics is identical: chain rule applied to each scalar in the tensor. But tensor engines:
- Use matrix calculus (Jacobians, not scalar derivatives)
- Leverage BLAS/cuBLAS for hardware-accelerated matrix operations
- Process an entire mini-batch simultaneously
- Can run on GPUs with CUDA

Our scalar engine is pedagogically clearer but ~10,000× slower than PyTorch for real networks. The key insight is that the LOGIC is the same — only the data representation differs.

## 7. Why Adam Usually Wins

- **SGD**: Same learning rate for all parameters → slow convergence when gradients differ by orders of magnitude.
- **Momentum**: Faster on smooth loss surfaces, but still same lr per param.
- **Adam**: Each parameter has its own effective learning rate: `lr / (sqrt(variance_estimate) + eps)`. Parameters with large gradient variance get smaller steps (cautious), parameters with small variance get larger steps (aggressive). This is approximately a diagonal approximation to Newton's method.
