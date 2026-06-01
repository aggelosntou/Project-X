"""
dataloader.py — Multi-worker DataLoader with prefetching.

The DataLoader wraps a Dataset and produces batches of (X, y) arrays.
It supports:
  - shuffle: randomise sample order each epoch
  - num_workers=0: single-process, no IPC overhead
  - num_workers>0: multiprocessing with a prefetch queue

Design notes
------------
The multiprocessing implementation spawns a new process per batch, which
is educational but inefficient for large datasets (spawn overhead dominates
for small batches).  Real frameworks (PyTorch) use persistent worker
processes with a shared result queue.  The key lesson here is the
architecture: main process yields batches while workers prefetch the next N.
"""

import numpy as np
import multiprocessing as mp
import time


# ---------------------------------------------------------------------------
# Worker function (must be module-level for multiprocessing spawn on macOS)
# ---------------------------------------------------------------------------

def _worker_fn(dataset, indices, result_queue, transform):
    """
    Worker process: load items for the given indices, apply transform,
    stack into arrays, and put the batch onto result_queue.

    This function runs in a separate process (no GIL contention).
    Data is communicated back via result_queue (pickle serialisation).
    """
    items = []
    for idx in indices:
        x, y = dataset[int(idx)]
        if transform is not None:
            x = transform(x)
        items.append((x, y))

    xs = np.stack([item[0] for item in items])
    ys = np.array([item[1] for item in items])
    result_queue.put((xs, ys))


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Multi-worker DataLoader with optional prefetching.

    Parameters
    ----------
    dataset        : Dataset instance
    batch_size     : int  (default 32)
    shuffle        : bool — randomise order each epoch (default True)
    num_workers    : int  — number of parallel worker processes (0 = single-threaded)
    prefetch_factor: int  — how many batches ahead to prefetch (only used when
                    num_workers > 0)
    transform      : callable or None — applied to each feature array
    drop_last      : bool — drop the last batch if it is smaller than batch_size

    Usage
    -----
        loader = DataLoader(dataset, batch_size=64, num_workers=2)
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            ...
    """

    def __init__(self, dataset, batch_size: int = 32, shuffle: bool = True,
                 num_workers: int = 0, prefetch_factor: int = 2,
                 transform=None, drop_last: bool = False):
        self.dataset        = dataset
        self.batch_size     = batch_size
        self.shuffle        = shuffle
        self.num_workers    = num_workers
        self.prefetch_factor = prefetch_factor
        self.transform      = transform
        self.drop_last      = drop_last

    def _make_batches(self):
        """Return list of index arrays, one per batch."""
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        batches = []
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start:start + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            batches.append(batch)
        return batches

    def __iter__(self):
        batches = self._make_batches()

        if self.num_workers == 0:
            # -----------------------------------------------------------------
            # Single-process path: simple, no overhead
            # -----------------------------------------------------------------
            for batch_indices in batches:
                items = []
                for idx in batch_indices:
                    x, y = self.dataset[int(idx)]
                    if self.transform is not None:
                        x = self.transform(x)
                    items.append((x, y))

                yield (np.stack([item[0] for item in items]),
                       np.array([item[1] for item in items]))

        else:
            # -----------------------------------------------------------------
            # Multi-process path: prefetch batches in parallel
            #
            # Architecture:
            #   - result_q: bounded queue shared between all workers and main
            #   - We launch up to (prefetch_factor * num_workers) workers ahead
            #   - As each batch is yielded, we launch the next worker
            #
            # Limitation: we spawn a new process per batch, which has O(10ms)
            # startup overhead.  Real dataloaders use persistent worker pools.
            # -----------------------------------------------------------------
            max_prefetch = self.prefetch_factor * self.num_workers
            result_q  = mp.Queue(maxsize=max_prefetch + 4)
            workers   = []
            batch_iter = iter(batches)
            n_batches  = len(batches)
            n_launched = 0
            n_yielded  = 0

            # Spawn initial wave of workers
            for _ in range(min(max_prefetch, n_batches)):
                try:
                    batch_idx = next(batch_iter)
                    p = mp.Process(
                        target=_worker_fn,
                        args=(self.dataset, batch_idx, result_q, self.transform),
                        daemon=True
                    )
                    p.start()
                    workers.append(p)
                    n_launched += 1
                except StopIteration:
                    break

            # Yield results and launch next worker for each yielded
            while n_yielded < n_batches:
                xs, ys = result_q.get()
                n_yielded += 1
                yield xs, ys

                # Launch next worker if more batches remain
                if n_launched < n_batches:
                    try:
                        batch_idx = next(batch_iter)
                        p = mp.Process(
                            target=_worker_fn,
                            args=(self.dataset, batch_idx, result_q, self.transform),
                            daemon=True
                        )
                        p.start()
                        workers.append(p)
                        n_launched += 1
                    except StopIteration:
                        pass

            # Wait for all workers to finish
            for p in workers:
                p.join()

    def __len__(self):
        """Number of batches per epoch."""
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from dataset import ArrayDataset

    rng = np.random.default_rng(0)
    X   = rng.standard_normal((500, 20)).astype(np.float32)
    y   = rng.integers(0, 5, 500)
    ds  = ArrayDataset(X, y)

    # --- Single-process test ---
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    total = 0
    for xb, yb in loader:
        total += len(xb)
    assert total == 500, f"Expected 500 samples, got {total}"
    print(f"Single-process: {total} samples, {len(loader)} batches — OK")

    # --- Multi-process test (small dataset; tests architecture) ---
    loader_mp = DataLoader(ds, batch_size=50, shuffle=False,
                           num_workers=2, prefetch_factor=2)
    total_mp = 0
    for xb, yb in loader_mp:
        total_mp += len(xb)
    assert total_mp == 500, f"Expected 500 samples, got {total_mp}"
    print(f"Multi-process:  {total_mp} samples, {len(loader_mp)} batches — OK")

    print("\nAll DataLoader self-tests passed.")
