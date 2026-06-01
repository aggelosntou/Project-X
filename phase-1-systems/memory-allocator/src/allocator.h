/*
 * allocator.h — Public interface for our custom free-list memory allocator.
 *
 * WHY build our own allocator?
 *   The system's malloc() is a black box. By building one ourselves, we learn
 *   exactly how heap memory is managed: how the OS gives us raw memory, how we
 *   carve it into user-visible chunks, and how we recycle those chunks on free.
 *
 * DESIGN OVERVIEW:
 *   We maintain a linked list of "blocks", each preceded by a small header.
 *   The header stores the block's size and whether it is currently free.
 *   This is called a "free list" allocator (or "implicit free list" because
 *   every block — free or not — is in the list).
 */

#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <stddef.h>  /* size_t */

/* ── Public API ─────────────────────────────────────────────────────────── */

/**
 * my_malloc(size) — Allocate `size` bytes of heap memory.
 *
 * Returns a pointer to the first usable byte (past the hidden header).
 * Returns NULL if allocation fails.
 *
 * WHY does it look like system malloc? Because we want a drop-in replacement
 * that exercises the same interface, so we can benchmark both.
 */
void *my_malloc(size_t size);

/**
 * my_free(ptr) — Return a previously-allocated block to the free list.
 *
 * ptr must be a pointer returned by my_malloc or my_realloc.
 * Passing NULL is a no-op (matches POSIX behavior).
 *
 * WHY can't we just forget about it? The OS gave us a big chunk of address
 * space; we track which pieces are in use so we can reuse them.
 */
void my_free(void *ptr);

/**
 * my_realloc(ptr, size) — Resize an existing allocation.
 *
 * Handles corner cases:
 *   ptr == NULL  → same as my_malloc(size)
 *   size == 0    → same as my_free(ptr), returns NULL
 *   otherwise    → grow or shrink, possibly moving data
 */
void *my_realloc(void *ptr, size_t size);

/**
 * allocator_dump() — Print the entire heap free list for debugging.
 *
 * For each block, prints: address, size, whether it is free or allocated.
 * Indispensable when debugging heap corruption or fragmentation.
 */
void allocator_dump(void);

#endif /* ALLOCATOR_H */
