"""
dataset.py — Dataset abstractions for the data-pipeline project.

Provides:
  - Dataset      : abstract base class
  - ArrayDataset : wrap NumPy arrays
  - CSVDataset   : lazy-load rows from a CSV file with O(1) random access
  - CachedDataset: LRU-cache wrapper over any Dataset
"""

import numpy as np
import os


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Dataset:
    """
    Abstract dataset interface.

    Subclasses must implement __len__ and __getitem__.
    __getitem__ should return a (features, label) tuple where features is a
    NumPy array and label is a scalar.
    """

    def __len__(self):
        raise NotImplementedError("Subclasses must implement __len__")

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__")

    def __repr__(self):
        return f"{self.__class__.__name__}(n={len(self)})"


# ---------------------------------------------------------------------------
# ArrayDataset
# ---------------------------------------------------------------------------

class ArrayDataset(Dataset):
    """
    Wraps a pair of NumPy arrays (X, y) as a Dataset.

    Parameters
    ----------
    X : np.ndarray  shape (N, ...)  — feature matrix
    y : np.ndarray  shape (N,)      — labels
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert len(X) == len(y), (
            f"X and y must have the same length, got {len(X)} and {len(y)}")
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# CSVDataset
# ---------------------------------------------------------------------------

class CSVDataset(Dataset):
    """
    Lazy-load rows from a CSV file with O(1) random access.

    Rather than loading the whole file into memory, this class:
    1. Reads the header and records the byte offset of every data row.
    2. On __getitem__, seeks to the exact byte offset and parses only that row.

    This is useful when the CSV is larger than RAM, or when only a subset
    of rows will be accessed per epoch.

    Parameters
    ----------
    filepath   : str — path to the CSV file
    target_col : int — column index of the label (default: -1, last column)
    delimiter  : str — field separator (default: ',')
    skip_header: bool — whether the first line is a header (default: True)
    """

    def __init__(self, filepath: str, target_col: int = -1,
                 delimiter: str = ',', skip_header: bool = True):
        self.filepath   = filepath
        self.target_col = target_col
        self.delimiter  = delimiter

        # Scan file once to record byte offsets of each data row
        self._offsets = []
        self._header  = None

        with open(filepath, 'rb') as f:
            if skip_header:
                self._header = f.readline().decode('utf-8').strip()

            while True:
                offset = f.tell()
                line   = f.readline()
                if not line:
                    break
                stripped = line.strip()
                if stripped:          # skip blank lines
                    self._offsets.append(offset)

        self._n_rows = len(self._offsets)

    def __len__(self):
        return self._n_rows

    def __getitem__(self, idx):
        """
        Seek to the row at `idx` and parse it.

        Returns
        -------
        (features, label) where features is a float32 array and label is a float.
        """
        if idx < 0 or idx >= self._n_rows:
            raise IndexError(f"Index {idx} out of range [0, {self._n_rows})")

        with open(self.filepath, 'rb') as f:
            f.seek(self._offsets[idx])
            line   = f.readline().decode('utf-8').strip()

        values = [float(v) for v in line.split(self.delimiter)]
        values = np.array(values, dtype=np.float32)

        # Separate features and label
        n_cols = len(values)
        target_col = self.target_col % n_cols   # handle negative index

        label    = values[target_col]
        features = np.delete(values, target_col)
        return features, label

    def header(self):
        return self._header


# ---------------------------------------------------------------------------
# CachedDataset
# ---------------------------------------------------------------------------

class CachedDataset(Dataset):
    """
    LRU-cache wrapper over any Dataset.

    Keeps the `cache_size` most-recently-accessed items in memory.
    When the cache is full, the least-recently-used item is evicted.

    This is useful when:
    - The underlying dataset is slow to load (CSV, disk, network)
    - Items are accessed multiple times per epoch (e.g. small datasets
      trained for many epochs)

    Parameters
    ----------
    dataset    : Dataset — the wrapped dataset
    cache_size : int     — maximum number of items to keep in memory

    Notes
    -----
    The LRU order is maintained with a plain Python list, which has O(N)
    remove. For production use, replace with collections.OrderedDict
    for O(1) LRU operations.
    """

    def __init__(self, dataset: Dataset, cache_size: int = 1000):
        self._dataset    = dataset
        self._cache      = {}    # idx → item
        self._lru        = []    # access order (most recent at end)
        self._cache_size = cache_size

        self.n_hits   = 0
        self.n_misses = 0

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        if idx in self._cache:
            # Cache hit: move to most-recently-used position
            self._lru.remove(idx)
            self._lru.append(idx)
            self.n_hits += 1
            return self._cache[idx]

        # Cache miss: load from underlying dataset
        self.n_misses += 1
        item = self._dataset[idx]

        # Evict LRU item if cache is full
        if len(self._lru) >= self._cache_size:
            evict_idx = self._lru.pop(0)
            del self._cache[evict_idx]

        self._cache[idx] = item
        self._lru.append(idx)
        return item

    def cache_stats(self):
        total = self.n_hits + self.n_misses
        hit_rate = self.n_hits / total if total > 0 else 0.0
        return {
            "hits"     : self.n_hits,
            "misses"   : self.n_misses,
            "total"    : total,
            "hit_rate" : hit_rate,
            "cache_used": len(self._cache),
            "cache_size": self._cache_size,
        }

    def reset_stats(self):
        self.n_hits   = 0
        self.n_misses = 0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def create_synthetic_csv(filepath: str, n_rows: int = 1000,
                          n_features: int = 10, n_classes: int = 5,
                          seed: int = 0):
    """
    Write a synthetic CSV file for testing CSVDataset.

    The last column is the integer class label.
    """
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n_rows, n_features)).astype(np.float32)
    y   = rng.integers(0, n_classes, n_rows)

    header = ','.join([f'f{i}' for i in range(n_features)] + ['label'])

    with open(filepath, 'w') as f:
        f.write(header + '\n')
        for i in range(n_rows):
            row = ','.join([f'{v:.6f}' for v in X[i]] + [str(y[i])])
            f.write(row + '\n')

    return filepath


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(0)

    # --- ArrayDataset ---
    X = rng.standard_normal((100, 10)).astype(np.float32)
    y = rng.integers(0, 5, 100)
    ds = ArrayDataset(X, y)
    assert len(ds) == 100
    x0, y0 = ds[0]
    assert np.allclose(x0, X[0])
    print(f"ArrayDataset: len={len(ds)}, item shape={x0.shape}")

    # --- CSVDataset ---
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                     delete=False) as tmp:
        tmp_path = tmp.name

    create_synthetic_csv(tmp_path, n_rows=200, n_features=8)
    csv_ds = CSVDataset(tmp_path, target_col=-1)
    assert len(csv_ds) == 200
    feat, label = csv_ds[0]
    assert feat.shape == (8,)
    print(f"CSVDataset: len={len(csv_ds)}, feature shape={feat.shape}, label={label}")

    # Sequential access should agree
    feat50, _ = csv_ds[50]
    feat51, _ = csv_ds[51]
    feat50b, _ = csv_ds[50]   # re-access
    assert np.allclose(feat50, feat50b), "Re-accessing same row gives different result!"
    os.unlink(tmp_path)

    # --- CachedDataset ---
    cached = CachedDataset(ArrayDataset(X, y), cache_size=20)
    for i in range(50):
        _ = cached[i % 30]   # cycle to trigger eviction
    stats = cached.cache_stats()
    print(f"CachedDataset stats: {stats}")
    assert stats['hit_rate'] > 0, "Expected some cache hits"

    print("\nAll self-tests passed.")
