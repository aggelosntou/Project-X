/*
 * test_allocator.c — Test suite and benchmark for our custom allocator.
 *
 * Tests cover:
 *   1. Basic alloc/free correctness
 *   2. Fragmentation: many small allocs, free every other one, then a large alloc
 *   3. Realloc: shrink, grow in-place, grow with copy
 *   4. Timing vs system malloc (clock_gettime, CLOCK_MONOTONIC)
 */

#include "allocator.h"

#include <stdio.h>
#include <stdlib.h>   /* system malloc/free for comparison */
#include <string.h>
#include <time.h>
#include <assert.h>

/* ── Timing helpers ──────────────────────────────────────────────────────── */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ── Test 1: Basic allocation and free ───────────────────────────────────── */

static void test_basic(void) {
    printf("--- Test 1: Basic alloc/free ---\n");

    /* Allocate a small buffer and write to it */
    char *buf = my_malloc(64);
    assert(buf != NULL && "my_malloc returned NULL for 64 bytes");

    strcpy(buf, "Hello, allocator!");
    assert(strcmp(buf, "Hello, allocator!") == 0);

    /* Allocate a second buffer; they must not overlap */
    int *nums = my_malloc(10 * sizeof(int));
    assert(nums != NULL);
    for (int i = 0; i < 10; i++) nums[i] = i * i;
    assert(nums[9] == 81);

    /* The first buffer must still be intact */
    assert(strcmp(buf, "Hello, allocator!") == 0 && "Memory clobbered!");

    my_free(buf);
    my_free(nums);

    /* After freeing, we should be able to allocate the same sizes again
     * without the heap growing (free list reuse). */
    char *buf2 = my_malloc(64);
    assert(buf2 != NULL);
    my_free(buf2);

    /* NULL-free must be a no-op */
    my_free(NULL);

    printf("PASS\n\n");
}

/* ── Test 2: Fragmentation ───────────────────────────────────────────────── */

static void test_fragmentation(void) {
    printf("--- Test 2: Fragmentation + coalescing ---\n");

    /*
     * Allocate 20 small blocks, then free every other one.
     * Without coalescing, the freed blocks are isolated 32-byte holes.
     * With coalescing, adjacent freed blocks merge into larger chunks.
     */
    #define N_FRAGS 20
    void *ptrs[N_FRAGS];

    for (int i = 0; i < N_FRAGS; i++) {
        ptrs[i] = my_malloc(32);
        assert(ptrs[i] != NULL);
        /* Write a pattern so we can detect corruption */
        memset(ptrs[i], i, 32);
    }

    /* Free every other block (leave odd indices allocated) */
    for (int i = 0; i < N_FRAGS; i += 2) {
        my_free(ptrs[i]);
    }

    printf("Heap after freeing every other block:\n");
    allocator_dump();

    /* Now free the remaining blocks — coalescing should merge everything */
    for (int i = 1; i < N_FRAGS; i += 2) {
        my_free(ptrs[i]);
    }

    printf("Heap after freeing all blocks (coalesced):\n");
    allocator_dump();

    /* Now a large allocation should succeed by reusing coalesced space */
    void *big = my_malloc(32 * N_FRAGS);   /* same total as all fragments */
    assert(big != NULL && "Large alloc failed — coalescing may be broken");
    my_free(big);

    printf("PASS\n\n");
    #undef N_FRAGS
}

/* ── Test 3: Realloc ─────────────────────────────────────────────────────── */

static void test_realloc(void) {
    printf("--- Test 3: Realloc ---\n");

    /* realloc(NULL, n) == malloc(n) */
    int *arr = my_realloc(NULL, 5 * sizeof(int));
    assert(arr != NULL);
    for (int i = 0; i < 5; i++) arr[i] = i;

    /* Grow: data must be preserved */
    arr = my_realloc(arr, 10 * sizeof(int));
    assert(arr != NULL);
    for (int i = 0; i < 5; i++)
        assert(arr[i] == i && "Data lost during realloc grow");

    /* Shrink: data up to new size preserved */
    arr = my_realloc(arr, 3 * sizeof(int));
    assert(arr != NULL);
    for (int i = 0; i < 3; i++)
        assert(arr[i] == i && "Data lost during realloc shrink");

    /* realloc(ptr, 0) == free(ptr) */
    void *result = my_realloc(arr, 0);
    assert(result == NULL);

    printf("PASS\n\n");
}

/* ── Test 4: Timing benchmark ────────────────────────────────────────────── */

#define BENCH_ITERS  100000
#define BENCH_SIZE   128

static void bench_allocator(const char *label,
                            void *(*alloc_fn)(size_t),
                            void  (*free_fn)(void *)) {
    void *ptrs[100];
    double t0 = now_sec();

    for (int iter = 0; iter < BENCH_ITERS; iter++) {
        /* Allocate 100 blocks */
        for (int i = 0; i < 100; i++)
            ptrs[i] = alloc_fn(BENCH_SIZE);

        /* Write to each so the compiler cannot optimise them away */
        for (int i = 0; i < 100; i++)
            ((char *)ptrs[i])[0] = (char)i;

        /* Free in reverse order (common pattern) */
        for (int i = 99; i >= 0; i--)
            free_fn(ptrs[i]);
    }

    double elapsed = now_sec() - t0;
    long   total   = (long)BENCH_ITERS * 100;
    printf("  %-18s  %8.3f ms  (%ld alloc+free pairs, %.1f ns each)\n",
           label, elapsed * 1000.0, total, elapsed * 1e9 / total);
}

static void test_timing(void) {
    printf("--- Test 4: Timing comparison ---\n");
    printf("  (Each iteration: 100 allocs of %d bytes + 100 frees, "
           "repeated %d times)\n\n", BENCH_SIZE, BENCH_ITERS);

    bench_allocator("my_malloc",    my_malloc,  my_free);
    bench_allocator("system malloc", malloc,    free);

    printf("\n  NOTE: Our allocator is simpler and slower than glibc/jemalloc.\n"
           "  glibc malloc uses size-class free lists, per-thread arenas,\n"
           "  and many other tricks we deliberately omit for clarity.\n\n");
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("========================================\n");
    printf("  Custom Allocator Test Suite\n");
    printf("========================================\n\n");

    test_basic();
    test_fragmentation();
    test_realloc();
    test_timing();

    printf("All tests passed.\n");
    return 0;
}
