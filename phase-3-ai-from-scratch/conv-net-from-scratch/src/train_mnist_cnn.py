"""
train_mnist_cnn.py — Train a CNN on MNIST from scratch using NumPy.

Architecture:
  Conv(1,8,3,pad=1) → ReLU → MaxPool(2)       [28×28 → 14×14, 8 channels]
  Conv(8,16,3,pad=1) → ReLU → MaxPool(2)      [14×14 → 7×7, 16 channels]
  Flatten                                       [16×7×7 = 784 features]
  Linear(784, 64) → ReLU
  Linear(64, 10)                                [10 digit classes]

Why this architecture works:
  - First conv layers detect low-level features (edges, corners)
  - Second conv layers combine edges into shapes (loops, lines)
  - Pooling provides spatial invariance and reduces computation
  - Dense layers combine spatial features into class decisions

Expected: >95% test accuracy after 5 epochs
Training time: ~10-20 minutes on CPU (NumPy im2col convolution)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import urllib.request
import gzip
import struct
import time

from conv_layer import Conv2d
from pooling import MaxPool2d
from flatten import Flatten
from network import Sequential, Linear, ReLU, Adam, cross_entropy_loss


# ── MNIST download and loading ─────────────────────────────────────────────────

MNIST_URLS = {
    'train_images': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz',
    'test_images':  'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels':  'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz',
}

MNIST_FALLBACK_URLS = {
    'train_images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'test_images':  'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels':  'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
}


def download_mnist(data_dir: str = './data/mnist'):
    """Download MNIST dataset if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    files = {}

    for name, url in MNIST_URLS.items():
        filename = url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        files[name] = filepath

        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"  Saved to {filepath}")
            except Exception as e:
                print(f"  Primary URL failed: {e}")
                fallback = MNIST_FALLBACK_URLS[name]
                print(f"  Trying fallback: {fallback}")
                try:
                    urllib.request.urlretrieve(fallback, filepath)
                    print(f"  Saved to {filepath}")
                except Exception as e2:
                    print(f"  Fallback failed: {e2}")
                    raise RuntimeError(
                        f"Could not download {name}. "
                        "Please download MNIST manually and place .gz files in ./data/mnist/"
                    )
        else:
            print(f"  {filename}: already present")

    return files


def load_images(filepath: str) -> np.ndarray:
    """Read IDX image file (gzip compressed)."""
    with gzip.open(filepath, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(n, rows, cols)
    return images


def load_labels(filepath: str) -> np.ndarray:
    """Read IDX label file (gzip compressed)."""
    with gzip.open(filepath, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_mnist(data_dir: str = './data/mnist'):
    """Load and preprocess MNIST. Returns normalized float32 arrays."""
    files = download_mnist(data_dir)

    train_images = load_images(files['train_images']).astype(np.float32) / 255.0
    train_labels = load_labels(files['train_labels']).astype(np.int64)
    test_images  = load_images(files['test_images']).astype(np.float32) / 255.0
    test_labels  = load_labels(files['test_labels']).astype(np.int64)

    # Add channel dimension: (N, H, W) → (N, 1, H, W)
    train_images = train_images[:, np.newaxis, :, :]
    test_images  = test_images[:, np.newaxis, :, :]

    # Normalize to zero mean, unit variance
    mean = train_images.mean()
    std  = train_images.std()
    train_images = (train_images - mean) / (std + 1e-7)
    test_images  = (test_images  - mean) / (std + 1e-7)

    print(f"Train: {train_images.shape}, {train_labels.shape}")
    print(f"Test:  {test_images.shape},  {test_labels.shape}")
    return train_images, train_labels, test_images, test_labels


# ── Model definition ───────────────────────────────────────────────────────────

def build_model():
    """
    CNN for MNIST digit classification.

    Input: (N, 1, 28, 28)

    Conv(1→8, k=3, pad=1) → ReLU → MaxPool(2)    → (N,  8, 14, 14)
    Conv(8→16, k=3, pad=1) → ReLU → MaxPool(2)   → (N, 16,  7,  7)
    Flatten                                        → (N, 784)
    Linear(784→64) → ReLU                         → (N, 64)
    Linear(64→10)                                  → (N, 10)

    Parameters:
      Conv1: 8*(1*3*3 + 1)  =    80
      Conv2: 16*(8*3*3 + 1) =  1168
      FC1:   784*64 + 64    = 50240
      FC2:   64*10 + 10     =   650
      Total:                = 52,138  (vs MLP 28*28*256 = 200K+ for first layer alone)

    CNNs have far fewer parameters than MLPs because:
      1. Weight sharing: same filter reused at every spatial position
      2. Local connectivity: each neuron connects to a small patch, not all inputs
    """
    return Sequential([
        Conv2d(in_channels=1,  out_channels=8,  kernel_size=3, padding=1),
        ReLU(),
        MaxPool2d(kernel_size=2, stride=2),
        Conv2d(in_channels=8,  out_channels=16, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2d(kernel_size=2, stride=2),
        Flatten(),
        Linear(in_features=16 * 7 * 7, out_features=64),
        ReLU(),
        Linear(in_features=64, out_features=10),
    ])


# ── Training loop ──────────────────────────────────────────────────────────────

def evaluate(model: Sequential, images: np.ndarray, labels: np.ndarray,
             batch_size: int = 128) -> tuple:
    """Evaluate model accuracy and loss over a dataset."""
    n = len(labels)
    total_loss = 0.0
    n_correct  = 0

    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        x    = images[start:end]
        y    = labels[start:end]

        logits          = model.forward(x)
        loss, _         = cross_entropy_loss(logits, y)
        preds           = logits.argmax(axis=1)

        total_loss += float(loss) * len(y)
        n_correct  += int((preds == y).sum())

    avg_loss = total_loss / n
    accuracy = n_correct  / n
    return avg_loss, accuracy


def train():
    print("=" * 60)
    print("MNIST CNN TRAINING")
    print("=" * 60)

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.join(script_dir, '..', 'data', 'mnist')
    train_images, train_labels, test_images, test_labels = load_mnist(data_dir)

    # Build model
    model = build_model()
    print(model)
    n_params = sum(p.size for p, g in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")

    optimizer = Adam(lr=0.001)

    N_EPOCHS   = 5
    BATCH_SIZE = 64
    N_TRAIN    = len(train_labels)

    print(f"\nTraining for {N_EPOCHS} epochs, batch size {BATCH_SIZE}")
    print("-" * 60)
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Acc':>8} | {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, N_EPOCHS + 1):
        epoch_start = time.time()

        # Shuffle training data
        perm = np.random.permutation(N_TRAIN)
        train_images = train_images[perm]
        train_labels = train_labels[perm]

        train_loss = 0.0
        n_correct  = 0

        for start in range(0, N_TRAIN, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N_TRAIN)
            x   = train_images[start:end]
            y   = train_labels[start:end]

            # Forward
            logits = model.forward(x)

            # Loss
            loss, d_logits = cross_entropy_loss(logits, y)

            # Backward
            model.backward(d_logits)

            # Gradient clipping
            all_grads = np.concatenate([g.ravel() for p, g in model.parameters()])
            norm = np.sqrt((all_grads ** 2).sum())
            if norm > 5.0:
                scale = 5.0 / (norm + 1e-8)
                for p, g in model.parameters():
                    g *= scale

            # Optimizer step
            optimizer.step(model.parameters())
            model.zero_grad()

            train_loss += float(loss) * (end - start)
            n_correct  += int((logits.argmax(axis=1) == y).sum())

        train_loss /= N_TRAIN
        train_acc   = n_correct / N_TRAIN

        # Evaluate on test set
        _, test_acc = evaluate(model, test_images, test_labels)

        elapsed = time.time() - epoch_start
        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.2%} | {test_acc:>7.2%} | {elapsed:>5.1f}s")

    print("-" * 60)

    # Final evaluation
    _, final_test_acc = evaluate(model, test_images, test_labels)
    print(f"\nFinal test accuracy: {final_test_acc:.2%}")

    if final_test_acc >= 0.95:
        print("SUCCESS: Reached target accuracy >= 95%")
    else:
        print(f"Note: {final_test_acc:.2%} < 95% target. Try more epochs or check implementation.")

    return model


if __name__ == "__main__":
    np.random.seed(42)
    train()
