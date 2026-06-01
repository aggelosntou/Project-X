/*
 * allocator.c — Free-list heap allocator built on top of sbrk().
 *
 * ── HOW THE HEAP WORKS ────────────────────────────────────────────────────
 *
 *  Virtual address space (simplified):
 *
 *    [ text | data | bss | HEAP → ... ← STACK ]
 *                           ↑
 *                      program break
 *
 *  sbrk(n) moves the "program break" — the boundary between the heap and
 *  unmapped memory — forward by n bytes, giving us fresh zeroed memory.
 *  It returns the OLD break (i.e., the start of the newly-added region).
 *
 *  We never call sbrk() with negative values here (shrinking the heap back
 *  to the OS is possible but tricky; real allocators rarely do it).
 *
 * ── BLOCK LAYOUT ─────────────────────────────────────────────────────────
 *
 *   ┌──────────────────────────────────────┬──────────────────────────────┐
 *   │            Block Header              │       User payload           │
 *   │  size_t size | int is_free | *next   │   (size bytes)               │
 *   └──────────────────────────────────────┴──────────────────────────────┘
 *   ↑                                      ↑
 *   internal pointer (Block*)              what my_malloc returns
 *
 *  'size' is the payload size, NOT including the header.
 *  'next' links to the next block header (free or not).
 *
 * ── ALLOCATION STRATEGY: FIRST FIT ───────────────────────────────────────
 *
 *  Walk the list from head; return the first free block that is large enough.
 *  This is O(n) but simple and good enough for teaching purposes.
 *  Best-fit would reduce external fragmentation but is slower.
 *
 * ── SPLITTING ─────────────────────────────────────────────────────────────
 *
 *  If a free block is larger than needed, we split it:
 *
 *   Before:  [ hdr | ........... free ........... ]
 *   After:   [ hdr | used ][ hdr | .... free .... ]
 *
 *  This reduces internal fragmentation (wasted space inside a block).
 *
 * ── COALESCING ────────────────────────────────────────────────────────────
 *
 *  When we free a block, we merge it with any immediately-adjacent free
 *  blocks. Without coalescing, repeated malloc/free of same-size blocks
 *  causes external fragmentation: plenty of total free space but no single
 *  contiguous chunk large enough to satisfy a big request.
 */

#include "allocator.h"

#include <stdio.h>
#include <string.h>   /* memcpy */
#include <unistd.h>   /* sbrk   */

/* ── Block header ────────────────────────────────────────────────────────── */

typedef struct Block {
    size_t        size;     /* payload size (bytes, NOT including this header) */
    int           is_free;  /* 1 = available for reuse, 0 = in use */
    struct Block *next;     /* pointer to the next header in the list */
} Block;

/*
 * HEADER_SIZE — how many bytes the header occupies.
 *
 * WHY not just sizeof(Block)?  On most 64-bit systems sizeof(Block) will be
 * 24 bytes already aligned, but we keep it explicit and aligned to 8 bytes
 * so that the payload that follows us is also at least 8-byte aligned.
 * Misaligned access can crash on some architectures and is slow on x86.
 */
#define ALIGN8(x)    (((x) + 7) & ~(size_t)7)
#define HEADER_SIZE  (ALIGN8(sizeof(Block)))

/* Minimum split threshold: only split if the leftover is large enough to be
 * useful (payload + header).  Splitting a 1-byte leftover wastes space. */
#define MIN_SPLIT    (HEADER_SIZE + 8)

/* ── Internal state ──────────────────────────────────────────────────────── */

/* head of the intrusive linked list; NULL before first allocation */
static Block *heap_head = NULL;

/* ── Helpers ─────────────────────────────────────────────────────────────── */

/*
 * block_from_ptr — given a user pointer (what malloc returned), recover
 * the Block header that sits just before it.
 *
 * Casting through char* before subtracting ensures byte-level arithmetic.
 */
static Block *block_from_ptr(void *ptr) {
    return (Block *)((char *)ptr - HEADER_SIZE);
}

/*
 * payload_from_block — given a Block header, return the pointer the user sees.
 */
static void *payload_from_block(Block *b) {
    return (char *)b + HEADER_SIZE;
}

/*
 * extend_heap — ask the OS for more memory via sbrk().
 *
 * We request HEADER_SIZE + size bytes, write a Block header at the start,
 * and return it.  Returns NULL on failure.
 *
 * WHY sbrk() and not mmap()?  sbrk() is the traditional, simpler approach
 * that makes the "program break" concept concrete.  Real allocators (jemalloc,
 * tcmalloc) use mmap() for large allocations because it allows returning
 * memory to the OS more easily.
 */
static Block *extend_heap(size_t size) {
    /* sbrk returns (void *)-1 on failure */
    void *raw = sbrk((intptr_t)(HEADER_SIZE + size));
    if (raw == (void *)-1) return NULL;

    Block *b  = (Block *)raw;
    b->size   = size;
    b->is_free = 0;  /* caller will use it immediately */
    b->next   = NULL;
    return b;
}

/*
 * split_block — if block `b` has more space than `needed`, carve a new
 * free block out of the tail.
 *
 * Before:  [b: size=100, used]
 * After:   [b: size=needed, used] → [tail: size=100-needed-HEADER_SIZE, free]
 */
static void split_block(Block *b, size_t needed) {
    size_t leftover = b->size - needed;
    if (leftover < MIN_SPLIT) return;   /* not worth splitting */

    /* carve the tail block out of b's payload area */
    Block *tail  = (Block *)((char *)payload_from_block(b) + needed);
    tail->size   = leftover - HEADER_SIZE;
    tail->is_free = 1;
    tail->next   = b->next;

    b->size = needed;
    b->next = tail;
}

/*
 * coalesce — after marking b as free, merge with its successor if also free.
 *
 * WHY only forward?  Our list is singly-linked (no 'prev' pointer) so we
 * can only look forward easily.  A doubly-linked list would let us coalesce
 * backwards too (boundary-tag / Knuth allocator), halving fragmentation in
 * some workloads.
 */
static void coalesce(Block *b) {
    while (b->next && b->next->is_free) {
        /* absorb the next block: add its header + payload to b's size */
        b->size += HEADER_SIZE + b->next->size;
        b->next  = b->next->next;
    }
}

/* ── Public API ─────────────────────────────────────────────────────────── */

void *my_malloc(size_t size) {
    if (size == 0) return NULL;

    /* Always allocate at least 8 bytes, and align the size so that every
     * header that follows us is also naturally aligned. */
    size = ALIGN8(size);

    /* ── First-fit search ────────────────────────────────────────────────
     * Walk every block (free or not).  Stop at the first free one that is
     * big enough. */
    Block *prev = NULL;
    Block *curr = heap_head;

    while (curr) {
        if (curr->is_free && curr->size >= size) {
            /* Found a fit.  Split it if there's enough leftover. */
            split_block(curr, size);
            curr->is_free = 0;
            return payload_from_block(curr);
        }
        prev = curr;
        curr = curr->next;
    }

    /* ── No fit found: extend the heap ──────────────────────────────────*/
    Block *new_block = extend_heap(size);
    if (!new_block) return NULL;   /* out of address space */

    /* Link the new block at the tail of our list */
    if (prev) {
        prev->next = new_block;
    } else {
        /* First allocation ever */
        heap_head = new_block;
    }

    return payload_from_block(new_block);
}

void my_free(void *ptr) {
    if (!ptr) return;   /* free(NULL) is defined to be a no-op */

    Block *b  = block_from_ptr(ptr);
    b->is_free = 1;

    /* Coalesce with any adjacent free blocks that follow us */
    coalesce(b);
}

void *my_realloc(void *ptr, size_t size) {
    /* realloc(NULL, size) == malloc(size) — POSIX requirement */
    if (!ptr) return my_malloc(size);

    /* realloc(ptr, 0) == free(ptr), return NULL — POSIX requirement */
    if (size == 0) {
        my_free(ptr);
        return NULL;
    }

    Block *b = block_from_ptr(ptr);
    size     = ALIGN8(size);

    /* ── In-place grow: if the current block is big enough, stay put ─── */
    if (b->size >= size) {
        /* Optionally split if we now have too much space */
        split_block(b, size);
        return ptr;
    }

    /*
     * ── In-place grow into adjacent free block ──────────────────────────
     * If the block immediately after us is free and the combined size is
     * sufficient, absorb it rather than allocating a new region and copying.
     *
     * This is an important optimisation: realloc is common when building
     * dynamic arrays, and avoiding memcpy for large buffers is a big win.
     */
    if (b->next && b->next->is_free &&
        (b->size + HEADER_SIZE + b->next->size) >= size) {

        b->size += HEADER_SIZE + b->next->size;
        b->next  = b->next->next;
        split_block(b, size);
        return ptr;
    }

    /* ── Fallback: allocate new, copy, free old ─────────────────────────*/
    void *new_ptr = my_malloc(size);
    if (!new_ptr) return NULL;

    /* Copy only the old payload (b->size bytes, not the new larger size) */
    memcpy(new_ptr, ptr, b->size < size ? b->size : size);
    my_free(ptr);
    return new_ptr;
}

void allocator_dump(void) {
    printf("\n=== Allocator Heap Dump ===\n");
    printf("%-20s  %-10s  %-8s\n", "Address", "Size", "Status");
    printf("%-20s  %-10s  %-8s\n", "-------", "----", "------");

    Block *b = heap_head;
    size_t total_free = 0, total_used = 0, block_count = 0;

    while (b) {
        printf("%-20p  %-10zu  %-8s\n",
               (void *)b, b->size,
               b->is_free ? "FREE" : "USED");

        if (b->is_free) total_free += b->size;
        else            total_used += b->size;
        block_count++;

        b = b->next;
    }

    printf("\nSummary: %zu blocks, %zu bytes used, %zu bytes free\n",
           block_count, total_used, total_free);

    /*
     * WHY print both?  The ratio of used to free reveals fragmentation.
     * If free >> 0 but malloc fails, you have external fragmentation:
     * lots of small free chunks that can't satisfy a large request.
     */
    printf("===========================\n\n");
}
