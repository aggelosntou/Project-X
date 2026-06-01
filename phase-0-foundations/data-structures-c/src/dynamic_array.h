/*
 * data-structures-c/src/dynamic_array.h
 *
 * A generic dynamic array (like C++ std::vector).
 *
 * Stores void * pointers so it can hold any type.
 * The caller is responsible for casting to/from void *.
 *
 * Growth strategy: double capacity when full.
 * This gives amortized O(1) push, O(1) get/set, O(n) for resize.
 */

#ifndef DYNAMIC_ARRAY_H
#define DYNAMIC_ARRAY_H

#include <stddef.h>

typedef struct {
    void  **data;   /* heap-allocated array of void pointers      */
    size_t  len;    /* number of items currently stored            */
    size_t  cap;    /* total allocated slots (always >= len)       */
} DynArray;

/* Create a new empty DynArray with the given initial capacity.
 * Returns a heap-allocated DynArray, or NULL on allocation failure. */
DynArray *da_new(size_t initial_cap);

/* Append item to the end.  Doubles capacity if full.
 * Returns 0 on success, -1 on allocation failure. */
int da_push(DynArray *da, void *item);

/* Return the item at index i.  Behaviour is undefined if i >= len.
 * (no bounds check in release mode — use da_len to guard yourself) */
void *da_get(const DynArray *da, size_t i);

/* Set the item at index i.  Undefined if i >= len. */
void da_set(DynArray *da, size_t i, void *item);

/* Remove and return the last item.  Returns NULL if empty. */
void *da_pop(DynArray *da);

/* Current number of items. */
size_t da_len(const DynArray *da);

/* Sort items using the provided comparator (same signature as qsort). */
void da_sort(DynArray *da, int (*cmp)(const void *, const void *));

/* Free all memory associated with da (including the DynArray struct itself).
 * Does NOT free the items pointed to by the void pointers — that's the
 * caller's responsibility. */
void da_free(DynArray *da);

#endif /* DYNAMIC_ARRAY_H */
