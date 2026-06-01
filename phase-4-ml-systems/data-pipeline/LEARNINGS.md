# Learnings — Data Pipeline

## 1. Why data loading is often the training bottleneck

In image classification, a single training sample requires:

1. **Disk read**: open the JPEG file (~50–200 KB) and read bytes → ~1–10 ms
   on an SSD, ~5–50 ms on an HDD.
2. **Decode**: decompress JPEG → ~1–5 ms (CPU-bound).
3. **Resize**: bilinear interpolation to target resolution → ~0.5 ms.
4. **Augment**: random crop, flip, colour jitter → ~1–2 ms.

A single CPU core can prepare ~100–300 images/second.  A modern GPU can
process >1 000 images/second in a forward pass.  Without overlapping
data preparation with GPU compute, the GPU is idle most of the time.

This is the fundamental motivation for multi-worker data loading.

---

## 2. Why prefetching overlaps IO with compute

Consider training with batch_size=64:

### Without prefetching (sequential):
```
[ Load batch 1 ][ Compute batch 1 ][ Load batch 2 ][ Compute batch 2 ] ...
     300 ms           100 ms             300 ms           100 ms
Total: 800 ms for 2 batches → 400 ms per batch
GPU utilisation: 100ms / 400ms = 25%
```

### With prefetching (1 batch ahead):
```
[ Load batch 1 ][ Load batch 2 | Compute batch 1 ][ Load batch 3 | Compute batch 2 ]
     300 ms           300 ms           300 ms
Total: ~600 ms for 2 batches → 300 ms per batch
GPU utilisation: 300ms / 300ms = 100%  (limited by IO)
```

Prefetching moves the CPU (IO + transform) off the GPU's critical path.
The GPU computes on batch N while the CPU prepares batch N+1.

PyTorch's `DataLoader` with `num_workers > 0` does exactly this:
workers continuously fill a queue; the training loop just reads from the queue.

---

## 3. Why too many workers can hurt

### a. Process spawn overhead (critical on macOS / Windows)

Python's `multiprocessing` on macOS uses the **spawn** start method:
each worker process forks a new Python interpreter and reimports all modules.
This takes ~50–200 ms per process.  For a small dataset where each epoch
takes 500 ms, spawning 4 workers costs 200–800 ms of overhead — more than
the actual work.

Linux can use **fork** (copies parent memory), which is much faster (~1 ms)
but can cause issues with threaded code (numpy, OpenBLAS) due to fork-safety.

### b. Memory copies via pickle serialisation

When a worker process sends a batch back to the main process, it must
**pickle** (serialise) the NumPy arrays, send them through a pipe or socket,
and the main process must **unpickle** them.  This involves:
- Memory copy: O(batch_size × feature_dim × itemsize)
- Pickle overhead: ~10–50 µs per call

For large batches or high-resolution images, this can become significant.

PyTorch avoids this with **shared memory** (`torch.multiprocessing`):
tensors live in shared memory and are accessed by both processes without
copying.  The worker only sends a handle (pointer), not the data.

### c. GIL contention (threading)

Python's **Global Interpreter Lock (GIL)** prevents two threads from
executing Python bytecode simultaneously.  For IO-bound work, the GIL is
released during `read()` calls, so threading helps.  But for CPU-bound
transforms (Python-level loops), threading provides no speedup.

This is why data loading uses **multiprocessing** (separate GILs) rather than
**threading**.

---

## 4. Multiprocessing vs threading

| Property | multiprocessing | threading |
|----------|----------------|-----------|
| Memory model | Separate address spaces | Shared address space |
| GIL | One per process (true parallelism) | Shared (no CPU parallelism for Python) |
| Communication | Pickle over pipe/socket | Direct memory access |
| Overhead | High (process spawn) | Low (thread creation) |
| Safety | Fork-unsafe with some libs | Race conditions if not careful |
| Best for | CPU-bound transforms | IO-bound waiting (network, disk) |

For NumPy-heavy transforms, `multiprocessing` wins because NumPy's C
extensions release the GIL during computation (e.g. `@` operator, `np.exp`).
Once the GIL is released, multiple threads could theoretically run in
parallel — but the overhead of thread scheduling and cache contention on
a single NUMA node often negates the benefit.

---

## 5. The CSVDataset: O(1) random access without loading the whole file

A naive CSV loader reads the entire file into memory at `__init__`.
This fails when:
- The file is larger than available RAM
- Only a small random subset of rows is needed per epoch

The `CSVDataset` implementation records the **byte offset** of each row
at init time (one sequential scan), then uses `file.seek(offset)` to jump
directly to any row on demand.  This is O(1) per random access vs O(N)
for a naive scan.

The offset table itself is O(N) ints (~8 bytes each), which is much smaller
than the actual data.  A 1 GB CSV file with 10 M rows needs ~80 MB for the
offset table vs 1 GB to load all data.

---

## 6. LRU cache for hot datasets

When the same items are accessed repeatedly (few epochs, small datasets),
an LRU cache avoids redundant disk reads.  The `CachedDataset` class wraps
any Dataset and keeps the most-recently-used `cache_size` items in memory.

Hit rate analysis:
- First epoch: 0% hit rate (cold cache, all misses)
- Second epoch with cache_size ≥ len(dataset): 100% hit rate
- With cache_size = len(dataset)/2 and random shuffling: ~50% hit rate

For image datasets, a cache of even 1 000 images (at 224×224×3×4 bytes
= ~600 MB) can meaningfully reduce IO, especially for repeated augmentations
of the same image.
