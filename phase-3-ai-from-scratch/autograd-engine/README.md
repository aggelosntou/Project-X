# Autograd Engine

A PyTorch-like automatic differentiation engine in pure Python — scalar-valued, dependency-free.

## Usage

```bash
cd src
python3 test_engine.py    # verify gradients are correct
python3 train_demo.py     # train MLP on XOR
```

## What is Automatic Differentiation?

Three ways to compute gradients:

1. **Symbolic**: `d/dx (x^2 + sin(x)) = 2x + cos(x)` — algebraic manipulation. Slow, produces complex expressions.
2. **Finite differences**: `f'(x) ≈ (f(x+ε) - f(x-ε)) / 2ε` — simple but slow (one pass per parameter) and imprecise.
3. **Automatic differentiation**: Record operations as they happen, then replay in reverse. Exact gradients, O(1) passes. This is what we implement.

## The Computation Graph

Every `Value` operation creates a node in a Directed Acyclic Graph (DAG):
```
L = (a + b) * c
         +           <-- node 1, parents = {a, b}
         * c         <-- node 2, parents = {+, c}
         = L
```

Edges point from output to inputs (backward direction).

## Topological Sort and Backward Pass

We can only compute `dL/d(parent)` after we know `dL/d(child)`. Topological sort guarantees we process every node AFTER all its consumers, so `.grad` is fully accumulated before we call `_backward()`.

## Files

- `src/engine.py` — `Value` class with full chain rule via closures
- `src/nn.py` — `Neuron`, `Layer`, `MLP` built on `Value`
- `src/optim.py` — `SGD`, `SGDMomentum`, `Adam`
- `src/train_demo.py` — XOR training demo
- `src/test_engine.py` — gradient verification vs analytical + PyTorch
