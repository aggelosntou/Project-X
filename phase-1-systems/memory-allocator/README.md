# memory-allocator

A from-scratch heap allocator in C using `sbrk()`, implementing the classic
free-list (first-fit) strategy with block splitting and coalescing.

## What it is

A drop-in replacement for `malloc` / `free` / `realloc` that makes the
internals of heap memory management visible and understandable. The code is
intentionally simple — real allocators add many layers of optimisation that
would obscure the core ideas.

## Files

| File | Purpose |
|------|---------|
| `src/allocator.h` | Public API declaration |
| `src/allocator.c` | Allocator implementation |
| `src/test_allocator.c` | Correctness tests + timing benchmark |
| `Makefile` | Build system |
| `LEARNINGS.md` | Conceptual explanations |
| `BENCHMARKS.md` | Expected performance numbers |

## How to build

```bash
make all
```

Requires a C11-capable compiler (`cc`, `clang`, or `gcc`). No external
dependencies — only POSIX (`unistd.h`, `time.h`).

## How to run tests

```bash
make test
```

This runs four tests:

1. **Basic alloc/free** — allocate buffers, write data, verify no overlap.
2. **Fragmentation** — allocate many blocks, free alternating ones, verify
   coalescing merges them so a large block can reuse the space.
3. **Realloc** — shrink, grow in-place, grow with copy, realloc-to-zero.
4. **Timing benchmark** — compare throughput against `system malloc`.

## Key concepts

See `LEARNINGS.md` for detailed explanations of `sbrk`, free lists,
fragmentation, and coalescing.
