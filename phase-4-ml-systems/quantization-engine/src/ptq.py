"""
ptq.py — Post-Training Quantization (PTQ) pipeline.

Steps:
  1. Define and "train" a small FP32 MLP (simulate training with random data)
  2. Calibrate activation quantization on a held-out calibration set
  3. Quantize all linear layers to INT8
  4. Evaluate accuracy before and after quantization

No PyTorch required — pure numpy.
"""

import sys
import time
import numpy as np

sys.path.insert(0, __file__.replace("ptq.py", ""))

from quantize import quantization_snr
from quantized_linear import QuantizedLinear


# ---------------------------------------------------------------------------
# FP32 MLP (the "pre-trained" model we will quantize)
# ---------------------------------------------------------------------------

class FP32MLP:
    """
    Simple MLP with ReLU activations.
    Weights are initialized as if the model has been trained (Xavier init).
    """

    def __init__(self, layer_sizes: list, rng: np.random.Generator):
        self.layer_sizes = layer_sizes
        self.weights     = []
        self.biases      = []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(6.0 / (n_in + n_out))
            W = rng.uniform(-limit, limit, (n_out, n_in)).astype(np.float32)
            b = np.zeros(n_out, dtype=np.float32)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ W.T + b
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0)          # ReLU
        return x

    def predict_class(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x)
        return np.argmax(logits, axis=1)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def n_params(self):
        return sum(W.size + b.size for W, b in zip(self.weights, self.biases))

    def memory_mb(self):
        return self.n_params() * 4 / 1e6

    def __repr__(self):
        return f"FP32MLP({self.layer_sizes})"


# ---------------------------------------------------------------------------
# Simple training loop (simulate "pre-training")
# ---------------------------------------------------------------------------

def _cross_entropy(probs, labels):
    n = labels.shape[0]
    return -np.mean(np.log(probs[np.arange(n), labels] + 1e-9))

def _softmax_cross_entropy_grad(probs, labels):
    n = labels.shape[0]
    grad = probs.copy()
    grad[np.arange(n), labels] -= 1
    return grad / n


def train_fp32_model(model, x_train, y_train, epochs=20, lr=0.01, batch_size=64,
                     rng=None):
    """
    Train the FP32 model with SGD + cross-entropy loss.
    This simulates a model that has already been trained before PTQ.
    """
    rng = rng or np.random.default_rng(0)
    n = x_train.shape[0]

    for epoch in range(epochs):
        idx = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            xb = x_train[batch_idx]
            yb = y_train[batch_idx]

            # --- Forward ---
            activations = [xb]
            h = xb
            for i, (W, b) in enumerate(zip(model.weights, model.biases)):
                z = h @ W.T + b
                if i < len(model.weights) - 1:
                    h = np.maximum(z, 0)
                else:
                    h = z
                activations.append(h)

            probs  = _softmax(activations[-1])
            loss   = _cross_entropy(probs, yb)
            epoch_loss += loss
            n_batches += 1

            # --- Backward ---
            delta = _softmax_cross_entropy_grad(probs, yb)
            for i in reversed(range(len(model.weights))):
                h_in = activations[i]
                dW = delta.T @ h_in / xb.shape[0]
                db = delta.mean(axis=0)
                model.weights[i] -= lr * dW
                model.biases[i]  -= lr * db
                if i > 0:
                    delta = delta @ model.weights[i]
                    delta *= (activations[i] > 0).astype(np.float32)

        if (epoch + 1) % 5 == 0:
            acc = evaluate_fp32(model, x_train, y_train)
            print(f"  Epoch {epoch+1:3d}: loss={epoch_loss/n_batches:.4f}  "
                  f"train_acc={acc:.3f}")


def _softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def evaluate_fp32(model, x, y_true):
    preds = model.predict_class(x)
    return float(np.mean(preds == y_true))


# ---------------------------------------------------------------------------
# PTQ pipeline
# ---------------------------------------------------------------------------

class PTQPipeline:
    """
    Post-Training Quantization pipeline.

    Usage:
        pipeline = PTQPipeline(fp32_model)
        pipeline.calibrate(calibration_data)
        q_model = pipeline.quantize(per_channel=True)
    """

    def __init__(self, fp32_model: FP32MLP):
        self.fp32_model = fp32_model
        self._calib_activations = []   # list of activation arrays per layer

    def calibrate(self, calib_data: np.ndarray, n_samples: int = 512):
        """
        Run the FP32 model on calibration data, recording activations
        at each layer input.  These are used to set the activation
        quantization parameters.
        """
        x = calib_data[:n_samples]
        self._calib_activations = [x.copy()]   # input to layer 0

        h = x
        for i, (W, b) in enumerate(zip(self.fp32_model.weights,
                                        self.fp32_model.biases)):
            z = h @ W.T + b
            if i < len(self.fp32_model.weights) - 1:
                h = np.maximum(z, 0)
            else:
                h = z
            self._calib_activations.append(h.copy())

        print(f"  Calibration: {len(self._calib_activations)} layer activations recorded "
              f"from {n_samples} samples.")

    def quantize(self, per_channel: bool = True,
                 calib_method: str = "minmax") -> list:
        """
        Convert all linear layers to INT8 QuantizedLinear layers.

        Returns a list of QuantizedLinear layers.
        """
        if not self._calib_activations:
            raise RuntimeError("Must call calibrate() before quantize().")

        q_layers = []
        for i, (W, b) in enumerate(zip(self.fp32_model.weights,
                                        self.fp32_model.biases)):
            ql = QuantizedLinear.from_fp32(W, b, per_channel=per_channel)
            # Feed the calibration activations for this layer
            ql.calibrate(self._calib_activations[i])
            ql.finalize_calibration(method=calib_method)
            q_layers.append(ql)
            print(f"  Layer {i}: {W.shape[1]} → {W.shape[0]}  "
                  f"act_scale={ql.act_scale:.6f}")

        return q_layers

    def quantized_forward(self, layers: list, x: np.ndarray) -> np.ndarray:
        for i, layer in enumerate(layers):
            x = layer(x)
            if i < len(layers) - 1:
                x = np.maximum(x, 0)
        return x

    def evaluate_quantized(self, q_layers: list, x: np.ndarray,
                            y_true: np.ndarray) -> float:
        logits = self.quantized_forward(q_layers, x)
        preds  = np.argmax(logits, axis=1)
        return float(np.mean(preds == y_true))


# ---------------------------------------------------------------------------
# Synthetic classification dataset
# ---------------------------------------------------------------------------

def make_classification_data(n_samples, n_features, n_classes, rng):
    """
    Generate a linearly-separable synthetic classification dataset.
    Each class is a Gaussian cluster.
    """
    centers = rng.standard_normal((n_classes, n_features)).astype(np.float32) * 3
    x_list, y_list = [], []
    per_class = n_samples // n_classes
    for c in range(n_classes):
        xi = centers[c] + rng.standard_normal((per_class, n_features)).astype(np.float32)
        x_list.append(xi)
        y_list.append(np.full(per_class, c, dtype=np.int64))
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    # Shuffle
    perm = rng.permutation(len(x))
    return x[perm], y[perm]


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(123)

    n_features = 64
    n_classes  = 10
    n_train    = 2000
    n_test     = 500
    n_calib    = 512

    layer_sizes = [n_features, 128, 64, n_classes]

    print("=" * 65)
    print("POST-TRAINING QUANTIZATION (PTQ) PIPELINE")
    print("=" * 65)

    # --- Dataset ---
    x_all, y_all = make_classification_data(n_train + n_test, n_features, n_classes, rng)
    x_train, y_train = x_all[:n_train], y_all[:n_train]
    x_test,  y_test  = x_all[n_train:], y_all[n_train:]
    x_calib          = x_train[:n_calib]

    # --- Build and train FP32 model ---
    print(f"\n[1] Training FP32 model {layer_sizes}  "
          f"({n_train} train, {n_test} test samples)")
    model = FP32MLP(layer_sizes, rng)
    train_fp32_model(model, x_train, y_train, epochs=30, lr=0.05, batch_size=64, rng=rng)

    acc_fp32_train = evaluate_fp32(model, x_train, y_train)
    acc_fp32_test  = evaluate_fp32(model, x_test,  y_test)
    print(f"\n  FP32 model — train acc: {acc_fp32_train:.3f}  "
          f"test acc: {acc_fp32_test:.3f}")
    print(f"  Model size: {model.memory_mb():.3f} MB ({model.n_params():,} params)")

    # --- PTQ ---
    print(f"\n[2] Calibrating on {n_calib} samples...")
    pipeline = PTQPipeline(model)
    pipeline.calibrate(x_calib, n_samples=n_calib)

    print(f"\n[3] Quantizing (per-channel INT8)...")
    q_layers_pc = pipeline.quantize(per_channel=True, calib_method="minmax")
    acc_int8_pc = pipeline.evaluate_quantized(q_layers_pc, x_test, y_test)

    print(f"\n[4] Quantizing (per-tensor INT8)...")
    pipeline.calibrate(x_calib, n_samples=n_calib)
    q_layers_pt = pipeline.quantize(per_channel=False, calib_method="minmax")
    acc_int8_pt = pipeline.evaluate_quantized(q_layers_pt, x_test, y_test)

    # --- Results ---
    int8_bytes = sum(ql.weight_memory_bytes()["int8_bytes"]
                     for ql in q_layers_pc)
    fp32_bytes = model.n_params() * 4

    print(f"\n{'=' * 65}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 65}")
    print(f"  {'Method':<22}  {'Test Acc':>9}  {'Model Size':>11}  {'Ratio':>7}")
    print(f"  {'-'*22}  {'-'*9}  {'-'*11}  {'-'*7}")
    print(f"  {'FP32':<22}  {acc_fp32_test:>9.3f}  "
          f"{fp32_bytes/1e3:>8.1f} KB  {1.0:>6.1f}x")
    print(f"  {'INT8 per-channel':<22}  {acc_int8_pc:>9.3f}  "
          f"{int8_bytes/1e3:>8.1f} KB  {fp32_bytes/int8_bytes:>6.1f}x")
    print(f"  {'INT8 per-tensor':<22}  {acc_int8_pt:>9.3f}  "
          f"{int8_bytes/1e3:>8.1f} KB  {fp32_bytes/int8_bytes:>6.1f}x")

    acc_drop_pc = acc_fp32_test - acc_int8_pc
    acc_drop_pt = acc_fp32_test - acc_int8_pt
    print(f"\n  Accuracy drop (per-channel): {acc_drop_pc:+.3f}")
    print(f"  Accuracy drop (per-tensor):  {acc_drop_pt:+.3f}")

    if abs(acc_drop_pc) < 0.02:
        print(f"\n  PTQ succeeded: <2% accuracy drop with 4x memory reduction.")
    else:
        print(f"\n  Note: Accuracy drop is significant — consider QAT.")

    # --- Latency comparison ---
    n_bench = 200
    t0 = time.perf_counter()
    for _ in range(n_bench):
        model.forward(x_test)
    fp32_time = (time.perf_counter() - t0) / n_bench * 1000

    t0 = time.perf_counter()
    for _ in range(n_bench):
        pipeline.quantized_forward(q_layers_pc, x_test)
    int8_time = (time.perf_counter() - t0) / n_bench * 1000

    print(f"\n  Inference latency ({n_test} samples):")
    print(f"    FP32 : {fp32_time:.3f} ms/batch")
    print(f"    INT8 : {int8_time:.3f} ms/batch")
    print(f"    (INT8 simulation includes numpy overhead;")
    print(f"     real INT8 hardware would be faster)")
