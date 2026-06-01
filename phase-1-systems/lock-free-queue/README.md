# Lock-Free Queue (Michael-Scott, 1996)

A fully-working educational implementation of a lock-free FIFO queue using
C11 atomics, along with a stress test and a head-to-head benchmark against
a mutex-protected queue.

## What this project covers

| Concept | Where to look |
|---|---|
| Michael-Scott algorithm | `src/lfq.c` — every step commented |
| CAS (compare-and-swap) | `src/lfq.h`, `src/lfq.c` |
| ABA problem | Block comment in `src/lfq.c` and `LEARNINGS.md` |
| Memory ordering (acquire/release) | Inline comments in `src/lfq.c` |
| Stress test (8 threads) | `src/test_lfq.c` |
| Lock-free vs mutex benchmark | `src/bench_lfq.c` |

## Build

```bash
make          # builds test_lfq and bench_lfq
```

Requires GCC with C11 support (`-std=c11`) and pthreads.
Tested on Linux x86-64 and macOS arm64 (Apple Silicon).

## Run the stress test

```bash
./test_lfq
```

Expected output (times will vary):

```
=== Lock-Free Queue Stress Test ===
Producers: 4  |  Consumers: 4  |  Items/producer: 250000
Total items: 1000000

--- Results ---
Total produced : 1000000
Total consumed : 1000000
PASS: all 1000000 items accounted for

Elapsed time : 0.1832 seconds
Throughput   : 5460000 ops/sec  (5.46 M ops/sec)
```

## Run the benchmark

```bash
./bench_lfq
```

See `BENCHMARKS.md` for an explanation of what results to expect and why.

## Key concepts (brief)

### Compare-and-Swap (CAS)

`CAS(addr, expected, desired)` is a single atomic CPU instruction that:
1. Reads `*addr`.
2. If it equals `expected`, writes `desired` and returns **true**.
3. Otherwise leaves `*addr` unchanged and returns **false**.

It is the primitive that allows lock-free progress: a thread that "loses"
simply retries rather than sleeping.

### The ABA Problem

See `LEARNINGS.md` for the full explanation.

### Memory Ordering

See `LEARNINGS.md` for acquire/release semantics explained from first principles.
