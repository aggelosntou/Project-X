# LEARNINGS — memory-allocator

## What does `sbrk()` do?

Every process has a *program break*: the virtual address marking the end of
the heap segment. Memory above this line is unmapped — accessing it causes a
segfault. `sbrk(n)` tells the kernel to extend the mapping by `n` bytes and
returns the old break (the start of the new region).

```
Before sbrk(1024):
  [ text | data | bss | heap ... ] | unmapped
                                  ↑
                            program break

After sbrk(1024):
  [ text | data | bss | heap ... | 1024 new bytes ] | unmapped
                                                    ↑
                                              new program break
```

`sbrk` is a thin wrapper around the `brk` syscall. Modern allocators prefer
`mmap(MAP_ANONYMOUS)` for large chunks because `mmap` can return individual
pages to the OS via `munmap`, whereas once you call `sbrk` you cannot easily
give the middle of the heap back.

## What is a free list?

A *free list* is a linked list of heap blocks. Each block is preceded by a
small metadata header. The list includes every block — free or in use — which
is why it is also called an *implicit free list*. An *explicit free list*
would link only the free blocks, making allocation faster (O(free blocks)
rather than O(all blocks)).

```
heap_head
   │
   ▼
┌────────────┬──────────┐    ┌────────────┬──────────┐    ┌────────────┬──────────┐
│ hdr(64,U)  │ payload  │───▶│ hdr(32,F)  │ (free)   │───▶│ hdr(128,U) │ payload  │ ──▶ NULL
└────────────┴──────────┘    └────────────┴──────────┘    └────────────┴──────────┘
  64 bytes, Used              32 bytes, Free                128 bytes, Used
```

## Internal vs external fragmentation

| Kind | Definition | Example | Fix |
|------|-----------|---------|-----|
| **Internal** | Wasted space *inside* an allocated block | malloc(1) returns 8-byte aligned block → 7 bytes wasted | Smaller alignment, size classes |
| **External** | Enough total free bytes but no single chunk is large enough | 3×32-byte free holes but you need 80 bytes | Coalescing adjacent free blocks |

Our allocator addresses:
- Internal fragmentation: align to 8 bytes (minimum) and split blocks
- External fragmentation: coalesce adjacent free blocks on `my_free`

## Why does coalescing matter?

Without coalescing, repeated allocation and deallocation of same-sized objects
fragments the heap into many small, isolated free regions. Consider:

```
1. malloc(64)  → block A
2. malloc(64)  → block B
3. free(A)     → [free 64] [used 64]
4. malloc(64)  → reuses A  ✓
5. free(B)     → [used 64] [free 64]   (A is now allocated again)
6. free(A)     → [free 64] [free 64]   ← without coalescing: TWO 64-byte holes
7. malloc(128) → FAIL!    ← even though 128 bytes total are free!
```

With coalescing, step 6 merges the two adjacent free blocks into one 128-byte
free block, and step 7 succeeds.

## Why first-fit vs best-fit?

- **First-fit**: scan from the head, return the first block that fits.
  Fast on average but can leave many small fragments at the front of the list.

- **Best-fit**: scan the entire list, return the smallest block that fits.
  Reduces wasted space inside the chosen block but is O(n) always, and
  produces many tiny unusable fragments over time.

- **Next-fit**: like first-fit but remembers where the last scan ended.
  Spreads fragmentation more evenly.

Real allocators (glibc ptmalloc, jemalloc, tcmalloc) use *size-class bins*:
separate free lists for blocks of sizes 8, 16, 32, 64, ... bytes. Allocation
is O(1): pick the bin for the requested size, pop the first element.

## What does alignment mean here?

Processors read memory most efficiently when the address is a multiple of the
data type's size. A `double` at address 7 requires two memory reads on most
architectures; at address 8 it takes one. Our `ALIGN8` macro rounds every
requested size up to the next multiple of 8, ensuring the payload pointer we
return satisfies 8-byte alignment requirements for all common types.
