"""
transforms.py — Data augmentation and preprocessing transforms.

Each transform is a callable that takes a NumPy array and returns a
transformed NumPy array.  Compose chains multiple transforms.

All transforms are stateless (except for seeding), so they are safe to
use from multiple processes/threads simultaneously.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class Normalize:
    """
    Subtract mean and divide by standard deviation (z-score normalisation).

    Parameters
    ----------
    mean : array-like — per-feature means (shape broadcastable to input)
    std  : array-like — per-feature standard deviations

    Notes
    -----
    Compute mean/std from the training set only (never the test set):
        mean = X_train.mean(axis=0)
        std  = X_train.std(axis=0)
    Then apply the same values to both train and test.
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std  = np.array(std,  dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-8)

    def __repr__(self):
        return f"Normalize(mean={self.mean.shape}, std={self.std.shape})"


class MinMaxNormalize:
    """
    Scale features to [0, 1] range using min/max of the training set.
    """

    def __init__(self, x_min, x_max):
        self.x_min = np.array(x_min, dtype=np.float32)
        self.x_max = np.array(x_max, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_min) / (self.x_max - self.x_min + 1e-8)

    def __repr__(self):
        return "MinMaxNormalize()"


# ---------------------------------------------------------------------------
# Augmentation: Flip
# ---------------------------------------------------------------------------

class RandomFlip:
    """
    Randomly flip the last two dimensions of the input (horizontal flip).

    Assumes input shape (..., H, W) or (C, H, W) — last axis is flipped.

    Parameters
    ----------
    prob : float — probability of flipping (default 0.5)
    axis : int   — axis to flip (default -1, i.e. last axis / horizontal)
    """

    def __init__(self, prob: float = 0.5, axis: int = -1):
        self.prob = prob
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.random() < self.prob:
            return np.flip(x, axis=self.axis).copy()
        return x

    def __repr__(self):
        return f"RandomFlip(prob={self.prob}, axis={self.axis})"


# ---------------------------------------------------------------------------
# Augmentation: Gaussian noise
# ---------------------------------------------------------------------------

class AddGaussianNoise:
    """
    Add zero-mean Gaussian noise to the input during training.

    This acts as a regulariser — it prevents memorisation of exact pixel
    values and improves generalisation (equivalent to L2 weight decay in
    certain linear models).

    Parameters
    ----------
    std : float — standard deviation of the noise (relative to input scale)
    """

    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x + np.random.randn(*x.shape).astype(x.dtype) * self.std

    def __repr__(self):
        return f"AddGaussianNoise(std={self.std})"


# ---------------------------------------------------------------------------
# Augmentation: Random crop / pad
# ---------------------------------------------------------------------------

class RandomCrop1D:
    """
    Randomly crop a 1D feature vector to a fixed output length.

    Useful for variable-length time series or audio features.

    Parameters
    ----------
    output_length : int — desired output length
    """

    def __init__(self, output_length: int):
        self.output_length = output_length

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if len(x) <= self.output_length:
            # Pad with zeros if shorter than output length
            pad = self.output_length - len(x)
            return np.pad(x, (0, pad))
        start = np.random.randint(0, len(x) - self.output_length + 1)
        return x[start:start + self.output_length]

    def __repr__(self):
        return f"RandomCrop1D(output_length={self.output_length})"


# ---------------------------------------------------------------------------
# Dtype conversion
# ---------------------------------------------------------------------------

class ToFloat32:
    """Cast any numeric array to float32."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.float32)

    def __repr__(self):
        return "ToFloat32()"


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

class Compose:
    """
    Chain multiple transforms in sequence.

    Parameters
    ----------
    transforms : list[callable] — applied left to right

    Usage
    -----
        t = Compose([ToFloat32(), Normalize(mean, std), AddGaussianNoise(0.05)])
        x_aug = t(x_raw)
    """

    def __init__(self, transforms: list):
        self.transforms = list(transforms)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        inner = ', '.join(repr(t) for t in self.transforms)
        return f"Compose([{inner}])"

    def __len__(self):
        return len(self.transforms)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100).astype(np.float32)

    # Normalize
    norm = Normalize(mean=x.mean(), std=x.std())
    xn = norm(x)
    print(f"Normalize: mean={xn.mean():.4f} (expect ~0), std={xn.std():.4f} (expect ~1)")

    # Noise
    noisy = AddGaussianNoise(std=0.1)(x)
    print(f"AddGaussianNoise: delta mean={abs((noisy - x).mean()):.4f}")

    # Flip 1D
    flip = RandomFlip(prob=1.0, axis=-1)
    xf = flip(x)
    assert np.allclose(xf, x[::-1]), "Flip failed"
    print("RandomFlip: passed")

    # Compose
    pipe = Compose([
        ToFloat32(),
        Normalize(mean=0.0, std=1.0),
        AddGaussianNoise(std=0.05),
    ])
    xout = pipe(x)
    print(f"Compose output shape: {xout.shape}, dtype: {xout.dtype}")

    print("\nAll self-tests passed.")
