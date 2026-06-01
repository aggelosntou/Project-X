# Benchmarks — Lock-Free vs Mutex Queue

## Workload

- 4 producer threads, each enqueuing 250,000 items = **1,000,000 total enqueues**
- 4 consumer threads sharing the dequeue work
- Items are integers cast to `void*` (no heap allocation per item on the
  consumer side)
- Timing: wall-clock time from thread launch to last thread join

## What to expect

### High-thread-count regime (this benchmark: 8 threads)

```
Lock-free : ~0.15 – 0.25 s
Mutex     : ~0.20 – 0.40 s
Speedup   : ~1.3 – 2.5x in favour of lock-free
```

Lock-free wins because threads never sleep/wake (no `futex` syscall) and
the CAS retry loop is very short.

### Low-thread-count regime (2 producers, 2 consumers)

```
Lock-free : ~0.10 s
Mutex     : ~0.08 s
Speedup   : mutex wins by ~1.2x
```

At low contention the mutex is almost never contested, so the fast-path
(`pthread_mutex_lock` without kernel entry) is cheaper than the CAS
spinning + cache-line invalidation of the lock-free algorithm.

## Key formulae

### Throughput

```
throughput (ops/sec) = total_items / elapsed_seconds
```

### Cache-coherency cost

Every successful CAS on a shared cache line forces all other cores to
invalidate their cached copy.  With `k` threads hammering `q->tail`:

```
coherency_messages_per_enqueue ≈ k - 1   (all other cores invalidate)
```

This is why throughput **does not scale linearly** with thread count.

### Why mutex overhead matters

A contested `pthread_mutex_lock` on Linux enters the kernel via `futex(2)`.
The round-trip cost is typically 1–5 µs.  With 1,000,000 items and heavy
contention, that adds up to:

```
extra_cost ≈ contention_rate × 1,000,000 × futex_latency
```

The lock-free queue avoids this entirely — at the cost of burning CPU
cycles in the CAS retry loop when many threads compete.

## Factors that affect results

| Factor | Favours lock-free | Favours mutex |
|---|---|---|
| Thread count | More threads | Fewer threads |
| CPU count / NUMA | Many cores | Single socket |
| Queue pressure | High (producers fast) | Low |
| OS scheduler | Preemptive (threads can be paused) | Cooperative |
| Critical section length | Short | Long |

## Recording your measurements

Replace the placeholders below after running `./bench_lfq`:

```
Platform  : _____________________________ (e.g. Apple M3, Linux x86-64)
OS        : _____________________________
Compiler  : gcc _____ / clang _____
Date      : _____________________________

Lock-free time : _______ s   (_______ M ops/sec)
Mutex time     : _______ s   (_______ M ops/sec)
Speedup        : _______x
```

## Profiling tips

To see where time is spent:

```bash
# Linux perf
perf stat -e cache-misses,cache-references,instructions ./bench_lfq

# macOS Instruments
instruments -t "Time Profiler" ./bench_lfq
```

Watch for high `cache-misses` in the lock-free run — that is the
cache-line ping-pong cost made visible.
