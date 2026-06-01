# Data Pipeline

A multi-worker data-loading pipeline built from scratch with NumPy and Python's
`multiprocessing` module.  The goal is to understand how production data loaders
(PyTorch `DataLoader`, TensorFlow `tf.data`) work under the hood.

## Project layout

```
data-pipeline/
├── src/
│   ├── dataset.py            # Dataset abstractions: Array, CSV, Cached (LRU)
│   ├── transforms.py         # Preprocessing and augmentation transforms
│   ├── dataloader.py         # Multi-worker DataLoader with prefetch queue
│   └── benchmark_pipeline.py # Throughput benchmark: num_workers 0/1/2/4
├── README.md
├── LEARNINGS.md
└── requirements.txt
```

## How to run

```bash
cd data-pipeline
pip install -r requirements.txt

# Run the throughput benchmark
python src/benchmark_pipeline.py

# Run self-tests for each module
python src/dataset.py
python src/transforms.py
python src/dataloader.py
```

## What this project builds

### `dataset.py`

| Class | Description |
|-------|-------------|
| `ArrayDataset` | Wraps a pair of NumPy arrays `(X, y)` |
| `CSVDataset` | Lazy-loads CSV rows; records byte offsets at init for O(1) random access |
| `CachedDataset` | LRU cache wrapper; tracks hit rate |

### `transforms.py`

| Transform | Description |
|-----------|-------------|
| `Normalize` | z-score: `(x - mean) / std` |
| `MinMaxNormalize` | Scale to `[0, 1]` |
| `RandomFlip` | Horizontal flip with probability p |
| `AddGaussianNoise` | Augmentation: adds N(0, std²) noise |
| `RandomCrop1D` | Random crop of 1D sequences |
| `Compose` | Chain transforms sequentially |

### `dataloader.py`

```python
from dataset import ArrayDataset
from dataloader import DataLoader

ds = ArrayDataset(X, y)
loader = DataLoader(ds, batch_size=64, num_workers=2, shuffle=True)

for X_batch, y_batch in loader:
    # X_batch: (64, n_features)  y_batch: (64,)
    ...
```

`num_workers=0` uses the main process (no IPC overhead).
`num_workers>0` spawns worker processes and prefetches batches into a queue.

## Expected benchmark output

```
--- Benchmark 1: No transform (pure array indexing) ---
  num_workers= 0  epoch_time=0.021s  samples/sec=476,190
  num_workers= 1  epoch_time=0.612s  samples/sec= 16,340
  num_workers= 2  epoch_time=0.714s  samples/sec= 14,006
  num_workers= 4  epoch_time=1.302s  samples/sec=  7,680
```

For small in-memory datasets, `num_workers=0` is fastest because process spawn
overhead (~50ms per process on macOS) dominates the actual work.

For real image datasets with disk IO and heavy augmentation, `num_workers=4`
can be 4–8x faster than `num_workers=0` due to IO/compute overlap.
