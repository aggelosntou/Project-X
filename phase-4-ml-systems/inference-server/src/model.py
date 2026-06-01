"""
model.py — Simple MLP model for the inference server.
Can be saved to / loaded from a numpy .npz checkpoint.
"""

import numpy as np
import os
import time


class MLP:
    """
    Multi-layer perceptron with ReLU hidden activations.

    Parameters
    ----------
    layer_sizes : list[int]  e.g. [128, 256, 64, 10]
    """

    def __init__(self, layer_sizes: list):
        self.layer_sizes = layer_sizes
        self.weights: list[np.ndarray] = []
        self.biases:  list[np.ndarray] = []

    @classmethod
    def random_init(cls, layer_sizes: list, seed: int = 42) -> "MLP":
        rng = np.random.default_rng(seed)
        m = cls(layer_sizes)
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(6.0 / (n_in + n_out))
            W = rng.uniform(-limit, limit, (n_out, n_in)).astype(np.float32)
            b = np.zeros(n_out, dtype=np.float32)
            m.weights.append(W)
            m.biases.append(b)
        return m

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x : shape (batch, in_features)
        returns : shape (batch, out_features)
        """
        x = np.asarray(x, dtype=np.float32)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ W.T + b
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0)   # ReLU
        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return softmax probabilities."""
        logits = self.forward(x)
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def save(self, path: str):
        arrays = {}
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"W{i}"] = W
            arrays[f"b{i}"] = b
        arrays["layer_sizes"] = np.array(self.layer_sizes)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "MLP":
        if not path.endswith(".npz"):
            path += ".npz"
        data = np.load(path)
        layer_sizes = data["layer_sizes"].tolist()
        m = cls(layer_sizes)
        n_layers = len(layer_sizes) - 1
        for i in range(n_layers):
            m.weights.append(data[f"W{i}"])
            m.biases.append(data[f"b{i}"])
        return m

    def n_params(self) -> int:
        return sum(W.size + b.size for W, b in zip(self.weights, self.biases))

    def __repr__(self):
        return f"MLP(layers={self.layer_sizes}, params={self.n_params():,})"


def get_or_create_model(checkpoint_path: str = "/tmp/inference_model.npz",
                        layer_sizes=None) -> MLP:
    """Load existing model or create a new random one."""
    if layer_sizes is None:
        layer_sizes = [64, 128, 64, 10]
    if os.path.exists(checkpoint_path):
        m = MLP.load(checkpoint_path)
        return m
    m = MLP.random_init(layer_sizes)
    m.save(checkpoint_path.replace(".npz", ""))
    return m


if __name__ == "__main__":
    m = MLP.random_init([64, 128, 64, 10])
    print(m)
    x = np.random.randn(8, 64).astype(np.float32)
    t0 = time.perf_counter()
    y = m.predict(x)
    t1 = time.perf_counter()
    print(f"Output shape: {y.shape}  (latency: {(t1-t0)*1000:.3f} ms for batch=8)")
    m.save("/tmp/test_mlp")
    m2 = MLP.load("/tmp/test_mlp.npz")
    assert np.allclose(m.predict(x), m2.predict(x))
    print("Save/load roundtrip OK.")
