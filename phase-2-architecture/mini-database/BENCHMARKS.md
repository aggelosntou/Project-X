# Mini Database — Expected Benchmarks

All measurements are approximate, on a modern laptop (Apple M2, NVMe SSD).
Your numbers will differ; the important takeaways are the ratios.

---

## B-Tree Insert

### Sequential inserts (id = 1, 2, 3, …)

| n rows   | total time | throughput     |
|----------|-----------|----------------|
| 1,000    | 0.1 ms    | ~10M inserts/s |
| 10,000   | 0.9 ms    | ~11M inserts/s |
| 100,000  | 9.3 ms    | ~10.7M inserts/s |
| 1,000,000| 97 ms     | ~10.3M inserts/s |

Sequential inserts are fast because:
- The B-tree only splits at the rightmost path; cache is warm.
- No page I/O during insert (all in memory until close).
- The B-tree height grows slowly: 1M rows in an order-3 tree ≈ 20 levels
  (much better with a real order-341 tree: ≈ 3 levels).

### Random inserts (id = random permutation of 1..n)

| n rows   | total time | vs sequential |
|----------|-----------|---------------|
| 1,000    | 0.1 ms    | ~1.0x         |
| 10,000   | 1.1 ms    | ~1.2x         |
| 100,000  | 12.1 ms   | ~1.3x         |
| 1,000,000| 148 ms    | ~1.5x         |

Random inserts are slightly slower because splits occur throughout the tree,
and cache locality is worse.  The gap would be much larger on a real disk-backed
tree due to random I/O (each node access could be a cache miss requiring a disk
read).

---

## B-Tree Search

### After inserting n rows

| n rows   | single lookup | 1000 lookups |
|----------|--------------|--------------|
| 1,000    | < 1 µs       | < 0.1 ms     |
| 10,000   | < 1 µs       | < 0.1 ms     |
| 100,000  | < 1 µs       | ~0.3 ms      |
| 1,000,000| ~2 µs        | ~2.0 ms      |

The lookup time grows logarithmically with n.  Doubling n adds at most one
more level to traverse.

---

## Search: B-Tree vs Linear Scan

For `SELECT * FROM rows WHERE id = 500000` with n = 1,000,000 rows:

| Method        | Time    | Notes                              |
|---------------|---------|------------------------------------|
| B-tree lookup | ~2 µs   | ~20 key comparisons (tree height)  |
| Linear scan   | ~50 ms  | must read every row from pages      |

**Ratio: ~25,000×** for a single lookup in 1M rows.

The scan time assumes in-memory pages (no disk I/O).  With disk I/O:
```
Linear scan: 1,000,000 rows * 48 bytes = ~11,750 pages = 11,750 disk reads
             At 100 µs per random read: ~1.2 seconds
B-tree:      ~20 page reads = 2 ms
Ratio:       ~600x (disk-bound)
```

For range queries (`WHERE id BETWEEN 100 AND 200`):
- B-tree: find id=100 in O(log n), then scan 100 entries = O(log n + 100).
- Linear scan: always O(n).
- B-tree wins for small ranges; linear scan wins only if the range is >~5%
  of the table (due to random vs sequential I/O patterns on spinning disks).

---

## Persistence

On close, the pager flushes all dirty pages to disk.

| Pages written | Flush time (NVMe) | Flush time (HDD) |
|---------------|-------------------|------------------|
| 1 (85 rows)   | ~0.1 ms           | ~5 ms            |
| 12 (1000 rows)| ~0.5 ms           | ~60 ms           |
| 118 (10K rows)| ~4 ms             | ~600 ms          |

NVMe SSD write bandwidth: ~3 GB/s sequential.
HDD write bandwidth: ~150 MB/s sequential, ~100 IOPS random.

For durability (fsync after each transaction), NVMe: ~100 µs per fsync.
This limits a durability-guaranteed database to ~10,000 committed transactions
per second per fsync call.  Postgres achieves ~50,000 TPS by batching
(group commit: one fsync per batch of N transactions).

---

## Physical Interpretation

### Why does B-tree height matter so much?

Each B-tree level is (potentially) one disk I/O.  At 5 ms per HDD seek:
```
height=3: 3 * 5 ms = 15 ms  per lookup
height=20: 20 * 5 ms = 100 ms per lookup
```

This is why database administrators make B-tree nodes as large as possible
(matching a disk sector or multiple of the OS page size) to minimise height.
SQLite uses 4 KiB pages; Postgres uses 8 KiB pages; some filesystems use
64 KiB or 1 MiB "cluster" sizes for large databases.

### What happens on a bulk load?

If you insert 1M rows in random order, every insert may touch O(log n) nodes
that are unlikely to be in the buffer cache.  This is called **B-tree thrashing**.

The fix: **bulk load** (sort the data first, then insert in sorted order).
Sorted inserts fill nodes sequentially from left to right, achieving ~100%
node utilisation vs ~70% for random inserts.  Postgres `COPY` and MySQL
`LOAD DATA` use this technique.
