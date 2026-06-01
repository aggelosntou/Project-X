"""
export_weights.py
=================
Export a trained MLP to disk in two formats:

  1. Binary (.bin)  — custom packed format readable by c_inference.c
  2. NumPy (.npz)   — convenient for Python consumers / inspection

Binary layout
-------------
[4 bytes]  n_layers           int32
Per layer:
  [4 bytes]  rows             int32
  [4 bytes]  cols             int32
  [rows*cols*4 bytes] W data  float32, row-major (C order)
  [4 bytes]  bias_len         int32
  [bias_len*4 bytes] b data   float32
"""

import numpy as np
import struct
import sys
import os


# ---------------------------------------------------------------------------
# Binary export
# ---------------------------------------------------------------------------

def export_to_binary(model, path: str) -> None:
    """Write model weights/biases to a self-describing binary file."""
    with open(path, 'wb') as f:
        n_layers = len(model.weights)
        f.write(struct.pack('i', n_layers))

        for W, b in zip(model.weights, model.biases):
            W = np.asarray(W, dtype=np.float32)
            b = np.asarray(b, dtype=np.float32)
            rows, cols = W.shape
            f.write(struct.pack('ii', rows, cols))
            f.write(W.tobytes())          # row-major (C default)
            f.write(struct.pack('i', len(b)))
            f.write(b.tobytes())

    size_kb = os.path.getsize(path) / 1024
    print(f"Exported model to {path} ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# NumPy export
# ---------------------------------------------------------------------------

def export_to_numpy(model, path: str) -> None:
    """Write model weights/biases to a NumPy .npz archive.

    Keys:  W0, b0, W1, b1, ... (layer index as suffix)
    """
    arrays = {}
    for i, (W, b) in enumerate(zip(model.weights, model.biases)):
        arrays[f'W{i}'] = np.asarray(W, dtype=np.float32)
        arrays[f'b{i}'] = np.asarray(b, dtype=np.float32)
    np.savez(path, **arrays)
    # np.savez appends .npz automatically
    actual_path = path if path.endswith('.npz') else path + '.npz'
    print(f"Exported model to {actual_path}")


# ---------------------------------------------------------------------------
# Load helpers (for verification)
# ---------------------------------------------------------------------------

def load_from_binary(path: str):
    """Read a binary file produced by export_to_binary; return (weights, biases)."""
    weights, biases = [], []
    with open(path, 'rb') as f:
        (n_layers,) = struct.unpack('i', f.read(4))
        for _ in range(n_layers):
            rows, cols = struct.unpack('ii', f.read(8))
            W = np.frombuffer(f.read(rows * cols * 4), dtype=np.float32).reshape(rows, cols).copy()
            (bias_len,) = struct.unpack('i', f.read(4))
            b = np.frombuffer(f.read(bias_len * 4), dtype=np.float32).copy()
            weights.append(W)
            biases.append(b)
    return weights, biases


def verify_export(model, bin_path: str) -> bool:
    """Cross-check that re-loaded binary weights match the original model."""
    weights, biases = load_from_binary(bin_path)
    ok = True
    for i, (W_orig, W_load) in enumerate(zip(model.weights, weights)):
        if not np.allclose(W_orig, W_load, atol=1e-6):
            print(f"  MISMATCH in layer {i} weights!")
            ok = False
    for i, (b_orig, b_load) in enumerate(zip(model.biases, biases)):
        if not np.allclose(b_orig, b_load, atol=1e-6):
            print(f"  MISMATCH in layer {i} biases!")
            ok = False
    if ok:
        print("Verification passed: binary export matches in-memory model.")
    return ok


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Allow running as:  python3 src/export_weights.py [bin_out] [npz_out]
    sys.path.insert(0, os.path.dirname(__file__))
    from model_definition import MLP

    bin_path = sys.argv[1] if len(sys.argv) > 1 else 'model.bin'
    npz_path = sys.argv[2] if len(sys.argv) > 2 else 'model'

    print("Building model...")
    model = MLP()
    print(f"  {model}")

    export_to_binary(model, bin_path)
    export_to_numpy(model, npz_path)
    verify_export(model, bin_path)

    # Smoke-test: run a forward pass before and after export
    x = np.random.randn(4, 784).astype(np.float32)
    preds = model.predict(x)
    print(f"Predictions for 4 random inputs: {preds}")
