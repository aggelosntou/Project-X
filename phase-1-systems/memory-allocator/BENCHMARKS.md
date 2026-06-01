# BENCHMARKS — memory-allocator

## Test setup

- Workload: 100 allocations of 128 bytes followed by 100 frees, repeated 100,000 times.
- Machine: Apple M-series or Intel x86-64, macOS, compiled with `-O2`.
- `clock_gettime(CLOCK_MONOTONIC)` for timing.

## Expected results

| Allocator | Time (ms) | ns per alloc+free |
|-----------|-----------|-------------------|
| `my_malloc` (this project) | ~80–200 ms | ~8–20 ns |
| system `malloc` (jemalloc/libSystem) | ~10–40 ms | ~1–4 ns |

Our allocator is 5–10× slower than the system allocator. This is expected
because:

1. **Linear scan**: First-fit walks the entire free list. jemalloc uses per-
   size-class lists for O(1) lookup.

2. **No caching**: Every free immediately returns the block to the list.
   tcmalloc keeps a per-thread cache of recently freed small blocks so the
   next malloc doesn't even need a lock.

3. **No thread safety**: Our allocator has a single global free list with no
   locking. Real allocators use per-thread arenas to avoid contention.

4. **sbrk overhead**: We call `sbrk` on every new extension. Real allocators
   request large slabs from the OS (via mmap) and subdivide them internally.

## What to look for when you run `make test`

- The `allocator_dump()` output after the fragmentation test should show a
  single large free block after all frees — evidence that coalescing works.
- The timing numbers will vary by machine but the ratio (our allocator vs
  system) is the important comparison.

## How to push performance

If you wanted to make this faster (as a learning exercise):

1. **Explicit free list**: link only free blocks → O(free) search instead of O(all).
2. **Size-class bins**: separate lists for 8, 16, 32, 64, 128 bytes → O(1).
3. **Bump allocator**: for short-lived allocations, just increment a pointer;
   reset all at once. Zero overhead but no individual frees.
4. **Arena allocator**: allocate from a fixed pool; good for game loops or
   request/response cycles with predictable lifetimes.
