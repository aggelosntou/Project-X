/*
 * data-structures-c/src/dynamic_array.c
 *
 * Implementation of the generic dynamic array.
 *
 * Key insight: we store void ** (pointer to pointer) as our data array.
 * Each slot holds one void * — a pointer to the actual item.
 * This lets us store any type without knowing its size.
 *
 * The trade-off: an extra indirection per access.
 * For small items (int, float), this has overhead.
 * For large items or heap-allocated objects, this is fine.
 */

#include "dynamic_array.h"
#include <stdlib.h>
#include <string.h>   /* memcpy */

#define DA_MIN_CAP 4  /* never allocate fewer than 4 slots */

DynArray *da_new(size_t initial_cap)
{
    DynArray *da = malloc(sizeof(DynArray));
    if (da == NULL) return NULL;

    /* Ensure at least DA_MIN_CAP to avoid constant reallocs at the start */
    size_t cap = initial_cap < DA_MIN_CAP ? DA_MIN_CAP : initial_cap;

    da->data = malloc(cap * sizeof(void *));
    if (da->data == NULL) {
        free(da);
        return NULL;
    }

    da->len = 0;
    da->cap = cap;
    return da;
}

int da_push(DynArray *da, void *item)
{
    /* If the array is full, grow it */
    if (da->len == da->cap) {
        /* Double the capacity.
         * WHY double? Because it guarantees amortized O(1) push.
         * If we grew by 1 each time, the N-th push would trigger N-1 copies
         * → total copies for N pushes = 0+1+2+...+(N-1) = O(N^2).
         * With doubling: copies = 1+2+4+...+N/2 < 2N = O(N) total. */
        size_t new_cap = da->cap * 2;
        void **new_data = realloc(da->data, new_cap * sizeof(void *));
        if (new_data == NULL) return -1;  /* allocation failed; da is still valid */
        da->data = new_data;
        da->cap  = new_cap;
    }

    da->data[da->len] = item;
    da->len++;
    return 0;
}

void *da_get(const DynArray *da, size_t i)
{
    /* No bounds check here — fast path.
     * In debug builds, you'd add: assert(i < da->len); */
    return da->data[i];
}

void da_set(DynArray *da, size_t i, void *item)
{
    da->data[i] = item;
}

void *da_pop(DynArray *da)
{
    if (da->len == 0) return NULL;
    da->len--;
    return da->data[da->len];
    /* We do NOT shrink the array on pop.
     * WHY: shrinking on every pop would make a push-pop-push-pop cycle
     * repeatedly hit the grow threshold → O(n) per operation.
     * Many implementations shrink when len < cap/4, but we keep it simple. */
}

size_t da_len(const DynArray *da)
{
    return da->len;
}

void da_sort(DynArray *da, int (*cmp)(const void *, const void *))
{
    /* qsort works on a contiguous array.  Our data IS contiguous (void **),
     * so we can sort the pointer array directly.
     * The comparator receives (void **) pointers (pointers to our void* slots).
     * Note: this sorts the POINTERS, not the pointed-to objects.
     * The caller's comparator should cast and dereference appropriately. */
    qsort(da->data, da->len, sizeof(void *), cmp);
}

void da_free(DynArray *da)
{
    if (da == NULL) return;
    free(da->data);
    /* Zero out the struct to catch use-after-free bugs early */
    da->data = NULL;
    da->len  = 0;
    da->cap  = 0;
    free(da);
}
