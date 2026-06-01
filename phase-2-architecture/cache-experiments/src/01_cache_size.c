/*
 * 01_cache_size.c — Detect cache hierarchy sizes by measuring access latency
 *
 * Technique: Iterate through an array of increasing size, reading every
 * element sequentially, and measure the time per access.
 *
 * When the working set fits in L1 cache: ~4 cycles/access (~1-2 ns)
 * When it spills to L2:                  ~12 cycles/access (~4 ns)
 * When it spills to L3:                  ~40 cycles/access (~15 ns)
 * When it spills to DRAM:                ~200+ cycles/access (~70+ ns)
 *
 * The "knee" in the latency curve reveals cache size boundaries.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* Get current time in nanoseconds */
static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* The array size must be large enough to exceed L3 cache.
 * Most laptops have L3 <= 32MB. We test up to 64MB. */
#define MAX_SIZE_BYTES  (64 * 1024 * 1024)  /* 64 MB */
#define ELEMENT_SIZE    sizeof(uint32_t)
#define MAX_ELEMENTS    (MAX_SIZE_BYTES / ELEMENT_SIZE)

/* Number of times to traverse the array — more iterations = better statistics */
#define ITERATIONS      4

/*
 * measure_latency — traverse `n` elements sequentially and return ns/element
 *
 * The "volatile" prevents the compiler from optimizing away the reads.
 * We sum into `sink` and print it at the end so the compiler can't elide the loop.
 */
static double measure_latency(uint32_t *arr, size_t n) {
    /* Warm up — ensure no cold-start effects */
    volatile uint64_t sink = 0;
    for (size_t i = 0; i < n; i++) sink += arr[i];

    uint64_t start = now_ns();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (size_t i = 0; i < n; i++) {
            sink += arr[i];
        }
    }
    uint64_t end = now_ns();

    /* Prevent dead-code elimination */
    if (sink == 0) printf("(sink=%llu)\n", (unsigned long long)sink);

    double total_accesses = (double)n * ITERATIONS;
    double elapsed_ns     = (double)(end - start);
    return elapsed_ns / total_accesses;
}

int main(void) {
    uint32_t *arr = malloc(MAX_SIZE_BYTES);
    if (!arr) {
        perror("malloc");
        return 1;
    }

    /* Initialize with non-zero data */
    for (size_t i = 0; i < MAX_ELEMENTS; i++) arr[i] = (uint32_t)(i + 1);

    printf("Cache Size Detection — Sequential Access Latency\n");
    printf("==================================================\n");
    printf("%-15s | %-12s | %-15s\n", "Array Size (KB)", "Elements", "ns/access");
    printf("%-15s-+-%-12s-+-%-15s\n", "---------------", "------------", "---------------");

    /* Test sizes: 1KB, 2KB, 4KB, 8KB, ... 64MB */
    for (size_t size_bytes = 1024; size_bytes <= MAX_SIZE_BYTES; size_bytes *= 2) {
        size_t n = size_bytes / ELEMENT_SIZE;
        double ns = measure_latency(arr, n);
        printf("%-15zu | %-12zu | %.2f\n", size_bytes / 1024, n, ns);
    }

    printf("\n");
    printf("Interpretation:\n");
    printf("  - Flat region at ~1-4 ns   → fits in L1 cache (~32-48 KB)\n");
    printf("  - Jump to ~4-15 ns         → L2 cache (~256 KB - 1 MB)\n");
    printf("  - Jump to ~15-40 ns        → L3 cache (~8-32 MB)\n");
    printf("  - Jump to ~50-100+ ns      → main memory (DRAM)\n");

    free(arr);
    return 0;
}
