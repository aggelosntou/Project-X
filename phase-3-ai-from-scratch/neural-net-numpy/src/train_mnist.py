"""
train_mnist.py — Download and train on MNIST. Target: >95% test accuracy.

MNIST: 60,000 training images, 10,000 test images, 28x28 grayscale, 10 classes.
Network: 784 → 256 → 128 → 10 (ReLU hidden, CrossEntropy loss)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import urllib.request
import gzip
import struct

np.random.seed(42)

from layers import Linear, ReLU
from losses import CrossEntropyLoss
from network import Sequential
from optimizers import Adam


# ------------------------------------------------------------------ #
#  MNIST download and parsing                                          #
# ------------------------------------------------------------------ #

MNIST_URL = "http://yann.lecun.com/exdb/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

def download_mnist(data_dir: str = "/tmp/mnist"):
    """Download MNIST files if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    for name, filename in MNIST_FILES.items():
        local = os.path.join(data_dir, filename)
        if not os.path.exists(local):
            url = MNIST_URL + filename
            print(f"  Downloading {url} ...")
            try:
                urllib.request.urlretrieve(url, local)
            except Exception as e:
                print(f"  Download failed: {e}")
                return None
    return data_dir


def load_images(path: str) -> np.ndarray:
    with gzip.open(path, 'rb') as f:
        magic, n, h, w = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, h * w).astype(np.float32) / 255.0


def load_labels(path: str) -> np.ndarray:
    with gzip.open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)


def load_mnist(data_dir: str = "/tmp/mnist"):
    X_train = load_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    y_train = load_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    X_test  = load_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    y_test  = load_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    return X_train, y_train, X_test, y_test


# ------------------------------------------------------------------ #
#  Confusion matrix                                                    #
# ------------------------------------------------------------------ #

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def print_confusion_matrix(cm: np.ndarray):
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print("     " + "".join(f"{i:5d}" for i in range(cm.shape[0])))
    for i, row in enumerate(cm):
        print(f"  {i}: " + "".join(f"{v:5d}" for v in row))


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    print("=== MNIST Training (NumPy) ===\n")

    # Try to load MNIST
    data_dir = "/tmp/mnist"
    if not all(os.path.exists(os.path.join(data_dir, f))
               for f in MNIST_FILES.values()):
        print("Downloading MNIST...")
        result = download_mnist(data_dir)
        if result is None:
            print("Download failed. Using synthetic data instead.")
            # Fallback: synthetic MNIST-like data
            n_train, n_test = 5000, 1000
            X_train = np.random.rand(n_train, 784).astype(np.float32)
            y_train = np.random.randint(0, 10, n_train)
            X_test  = np.random.rand(n_test, 784).astype(np.float32)
            y_test  = np.random.randint(0, 10, n_test)
            print("  (Note: random data, accuracy should be ~10%)")
        else:
            X_train, y_train, X_test, y_test = load_mnist(data_dir)
    else:
        X_train, y_train, X_test, y_test = load_mnist(data_dir)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Normalize: subtract mean, divide by std (per-feature)
    mean = X_train.mean(axis=0, keepdims=True)
    std  = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    # Model: 784 → 256 → 128 → 10
    model = Sequential([
        Linear(784, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 10),
    ])

    optimizer = Adam(model, lr=1e-3)
    criterion = CrossEntropyLoss()

    # Training
    n_epochs   = 20
    batch_size = 256
    n_train    = len(X_train)
    n_batches  = n_train // batch_size

    print(f"\nTraining: {n_epochs} epochs, batch_size={batch_size}")

    for epoch in range(n_epochs):
        # Shuffle
        perm = np.random.permutation(n_train)
        X_sh = X_train[perm]
        y_sh = y_train[perm]

        epoch_loss = 0.0
        for b in range(n_batches):
            xb = X_sh[b*batch_size:(b+1)*batch_size]
            yb = y_sh[b*batch_size:(b+1)*batch_size]

            logits = model.forward(xb)
            loss   = criterion.forward(logits, yb)
            epoch_loss += loss

            grad = criterion.backward()
            model.backward(grad)
            optimizer.step()

        # Test accuracy
        logits_test = model.forward(X_test)
        y_pred = logits_test.argmax(axis=1)
        acc = (y_pred == y_test).mean() * 100

        print(f"  Epoch {epoch+1:2d}/{n_epochs}: "
              f"loss={epoch_loss/n_batches:.4f}, "
              f"test_acc={acc:.2f}%")

        if acc > 97.0:
            print(f"\n  Achieved >97% accuracy at epoch {epoch+1}!")
            break

    # Final evaluation
    logits_test = model.forward(X_test)
    y_pred = logits_test.argmax(axis=1)
    final_acc = (y_pred == y_test).mean() * 100
    print(f"\nFinal test accuracy: {final_acc:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    print_confusion_matrix(cm)


if __name__ == '__main__':
    main()
