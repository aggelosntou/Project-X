import numpy as np


class MLP:
    """3-layer MLP: input -> 256 -> 128 -> 10

    Architecture:
        Layer 0: input_dim -> 256  (ReLU activation)
        Layer 1: 256       -> 128  (ReLU activation)
        Layer 2: 128       -> 10   (linear, logits out)

    Weights are initialised with Glorot uniform (Xavier) which keeps
    activations in a reasonable range regardless of network depth.
    """

    def __init__(self, layer_sizes=None):
        if layer_sizes is None:
            layer_sizes = [784, 256, 128, 10]
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        rng = np.random.default_rng(42)
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            W = rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)
            b = np.zeros(fan_out, dtype=np.float32)
            self.weights.append(W)
            self.biases.append(b)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x):
        """Return raw logits for input x (shape: [batch, input_dim] or [input_dim])."""
        x = np.asarray(x, dtype=np.float32)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ W + b
            if i < len(self.weights) - 1:
                x = np.maximum(0, x)  # ReLU
        return x  # logits

    def predict(self, x):
        """Return class indices (argmax of logits)."""
        logits = self.forward(x)
        return np.argmax(logits, axis=-1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def parameter_count(self):
        total = 0
        for W, b in zip(self.weights, self.biases):
            total += W.size + b.size
        return total

    def __repr__(self):
        shapes = " -> ".join(str(s) for s in self.layer_sizes)
        return f"MLP({shapes})  params={self.parameter_count():,}"
