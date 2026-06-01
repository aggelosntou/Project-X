"""
nn.py — Neural network modules built on the Value autograd engine.

This mirrors the design of PyTorch's nn.Module hierarchy:
  - Module: base class with parameters() method
  - Neuron: single neuron (dot product + bias + activation)
  - Layer: list of Neurons operating in parallel
  - MLP: stack of Layers

Every computation uses Value objects, so gradients flow automatically.
"""

import random
import math
from engine import Value


class Module:
    """Base class for all neural network modules."""

    def parameters(self):
        """Return a flat list of all learnable Value parameters."""
        return []

    def zero_grad(self):
        """Zero all parameter gradients before a backward pass."""
        for p in self.parameters():
            p.zero_grad()


class Neuron(Module):
    """
    A single artificial neuron:
        output = activation(w · x + b)

    where:
      w ∈ R^{n_in}  — weight vector
      b ∈ R         — scalar bias
      x ∈ R^{n_in}  — input vector
      w · x          — dot product = sum(w_i * x_i)
    """

    def __init__(self, n_in, activation='relu'):
        """
        Xavier initialization: weights ~ Uniform(-1/sqrt(n_in), 1/sqrt(n_in))

        Why Xavier?  We want the variance of the output to equal the variance of
        the input:  Var(w·x) = n_in * Var(w) * Var(x)
        Setting Var(w) = 1/n_in gives Var(output) ≈ Var(x).
        This prevents activations from vanishing or exploding as depth increases.
        """
        limit = 1.0 / math.sqrt(n_in)
        self.w = [Value(random.uniform(-limit, limit), label=f'w{i}')
                  for i in range(n_in)]
        self.b = Value(0.0, label='b')
        self.activation = activation

    def __call__(self, x):
        """Forward pass: compute activation(w·x + b)."""
        assert len(x) == len(self.w), f"Input size {len(x)} != weight size {len(self.w)}"

        # Dot product: sum of w_i * x_i + b
        # Python's sum(generator, start) accumulates the result
        pre_act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        # Apply activation
        if self.activation == 'relu':
            return pre_act.relu()
        elif self.activation == 'tanh':
            return pre_act.tanh()
        elif self.activation == 'sigmoid':
            return pre_act.sigmoid()
        elif self.activation == 'linear' or self.activation is None:
            return pre_act
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron(n_in={len(self.w)}, act={self.activation})"


class Layer(Module):
    """
    A fully-connected layer: n_out neurons, each taking n_in inputs.

    Mathematically: output ∈ R^{n_out} where output_j = Neuron_j(x)
    Together they compute: z = xW + b  (in matrix form, but here scalar-wise)
    """

    def __init__(self, n_in, n_out, activation='relu'):
        self.neurons = [Neuron(n_in, activation) for _ in range(n_out)]

    def __call__(self, x):
        """Evaluate all neurons and return list of outputs."""
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer([{', '.join(str(n) for n in self.neurons)}])"


class MLP(Module):
    """
    Multi-Layer Perceptron: a stack of fully-connected layers.

    Example: MLP(2, [4, 4, 1]) creates:
      Layer(2→4, relu) → Layer(4→4, relu) → Layer(4→1, linear)

    The last layer uses linear activation so we can apply any loss function
    without the output being constrained.
    """

    def __init__(self, n_in, layer_sizes, hidden_activation='relu'):
        sizes = [n_in] + layer_sizes
        # All hidden layers use hidden_activation; last layer is linear
        self.layers = []
        for i in range(len(layer_sizes)):
            activation = hidden_activation if i < len(layer_sizes) - 1 else 'linear'
            self.layers.append(Layer(sizes[i], sizes[i+1], activation))

    def __call__(self, x):
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        # If the last layer has a single output, unwrap from list
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP([{', '.join(str(l) for l in self.layers)}])"
