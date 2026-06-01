# Database Internals — Key Concepts

## 1. Why B-Trees?

A **B-tree of order m** is a search tree where every node holds up to m-1 keys
and up to m children.  Keys in a node divide its subtrees: all keys in
`children[i]` are between `keys[i-1]` and `keys[i]`.

### Complexity

| Operation | Array (unsorted) | Sorted array | Binary tree (worst) | B-tree |
|-----------|-----------------|--------------|---------------------|--------|
| Search    | O(n)            | O(log n)     | O(n)                | O(log n) |
| Insert    | O(1) amortised  | O(n)         | O(n)                | O(log n) |
| Delete    | O(n)            | O(n)         | O(n)                | O(log n) |

A balanced binary tree also achieves O(log n), so why B-trees?

### Fan-out minimises disk I/O

A binary tree with n=1,000,000 has height ~20.  A B-tree of order 500 has
height ~4.  Each node visit is one disk read.  Fewer levels = fewer disk reads.

Real databases use order 50–1000 depending on key size vs page size:
```
fan_out = PAGE_SIZE / (key_size + pointer_size)
        = 4096 / (4 + 8) = 341  (for 4-byte int keys and 8-byte pointers)
```

With fan_out=341 and n=10 million rows:
```
height = ceil(log_341(10,000,000)) = ceil(3.3) = 4 disk reads
```
vs. a binary tree: ceil(log_2(10,000,000)) = 24 disk reads.

That is a **6× reduction in I/O** for a common workload.

---

## 2. What Is a Page?

The operating system, disk controller, and CPU cache all work in fixed-size
chunks.  A **page** (4096 bytes on most modern systems) is the unit at which:

- The OS reads from / writes to disk (even if you ask for 1 byte, a full
  page is transferred).
- The OS page cache (buffer pool) manages memory.
- SSDs erase/write in "erase blocks" (typically 256 KiB = 64 pages).

**Consequence**: design your data structures to align to page boundaries.
If a B-tree node fits exactly in one page, each node access is exactly one
I/O operation.

```
Node size = key_size * MAX_KEYS + pointer_size * MAX_CHILDREN + metadata
          = 4*2 + 8*3 + 8 = 40 bytes  (our toy B-tree)
```

Our toy B-tree is tiny (order 3).  A real system would use a page-aligned
node of 4096 bytes, holding ~300 keys per node.

---

## 3. Index Scan vs Sequential Scan

**Sequential scan**: read every row, check predicate.
```
Time = n * (disk read time per row)
     = n * PAGE_SIZE / bandwidth
     ≈ 1,000,000 rows * 48 bytes / (500 MB/s) = ~96 ms
```

**Index scan** (B-tree): follow pointers from root to leaf.
```
Time = height * (page read time)
     = 4 * (4096 bytes / 500 MB/s)
     ≈ 0.03 ms  (1000× faster for n=1M)
```

The crossover point depends on the selectivity (fraction of rows returned):
- Returning 1 row → index always wins.
- Returning 50% of rows → sequential scan often wins (the index causes
  random I/O; sequential I/O can saturate disk bandwidth better).

PostgreSQL and MySQL switch to sequential scans when the query would touch
more than ~5–10% of the table (controlled by the query planner).

---

## 4. What Would ACID Require?

**ACID** = Atomicity, Consistency, Isolation, Durability.

Our toy database has none of these.  Here is what each would require:

### Atomicity
A transaction's changes are either all applied or none are.

Requires: **Write-Ahead Log (WAL)**.
1. Before modifying any page, write the intended change to a log file.
2. The log is `fsync`-ed to disk.
3. Then apply the change to the actual data pages.
4. On crash recovery, replay the log from the last checkpoint.

### Consistency
The database moves from one valid state to another.

Requires: constraint checking (foreign keys, NOT NULL, etc.) and rollback
on violation.

### Isolation
Concurrent transactions don't see each other's partial changes.

Requires: **MVCC** (Multi-Version Concurrency Control, used by Postgres)
or **lock-based concurrency** (used by older MySQL).  Both are complex.

### Durability
Committed transactions survive crashes.

Requires: `fsync()` (or `fdatasync()`) after writing the WAL entry.
`fsync` flushes the OS page cache to the physical disk.
Without it, a power failure after `write()` but before the disk commits
the data would lose the transaction.

Cost of `fsync`: ~1–10 ms per call on a spinning disk, ~100 µs on NVMe SSD.
This is why databases batch many transactions into one `fsync` (group commit).

---

## 5. Why SQLite Uses a B+ Tree Variant

SQLite stores both the index and the actual row data in a **B+ tree**:

- **B-tree** (our implementation): internal nodes store keys AND values.
- **B+ tree**: internal nodes store only keys (routing keys); all values
  are stored in leaf nodes, which are linked in a doubly-linked list.

Advantages of B+ tree:
1. **Range scans are fast**: scan the leaf chain without going back up to
   internal nodes.  `SELECT * FROM rows WHERE id BETWEEN 100 AND 200` is
   one B-tree traversal to find id=100, then a linked-list scan.
2. **Higher fan-out**: internal nodes hold more keys because they don't
   carry data, so the tree is shorter.
3. **Simpler caching**: only leaf pages contain data; internal pages are
   pure routing and can stay in cache indefinitely.

SQLite page types:
- **Interior pages**: routing keys + child page pointers.
- **Leaf pages**: keys + serialised row data (or index entries).
- **Overflow pages**: for rows too large to fit in one leaf page.

---

## 6. Buffer Pool Management

A production database's pager (buffer pool manager) must handle:

1. **Fixed-size cache** (e.g., 128 MB in SQLite; gigabytes in Postgres).
2. **LRU eviction**: when cache is full and a new page is needed, evict the
   Least Recently Used clean page.  If all pages are dirty, flush one first.
3. **Pin counts**: a page being actively used is "pinned" and cannot be evicted.
4. **Prefetching**: for sequential scans, fetch the next few pages before they
   are needed (hides I/O latency).

Our toy implementation has no eviction (it fits everything in RAM).

---

## 7. B-Tree Split and Merge (Visualised)

### Split (insert 50 into a full node [30, 70]):

```
Before:
  [30, 70]   (full: 2 keys = MAX_KEYS)

Insert 50 → triggers split:

After:
     [50]
    /    \
 [30]   [70]
```

The median key (50) is pushed up.  The left child keeps keys < 50,
the right keeps keys > 50.

### Merge (delete 50, both children have only 1 key):

```
Before:
     [50]
    /    \
 [30]   [70]

Delete 30 → left child empty → pull 50 down and merge:

After:
  [50, 70]   (leaf node, or stays internal if subtrees exist)
```

Merge is the reverse of split: the separator key is pulled down and the
two siblings combine into one node.
