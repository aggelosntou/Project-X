/*
 * 04_sequential_vs_random.c — Sequential vs random memory access
 *
 * With sequential access, hardware prefetchers detect the pattern and
 * preload data into cache ahead of time → nearly zero stall cycles.
 *
 * With random access, every access is an independent cache miss → the
 * CPU must wait ~100-200 ns for each DRAM fetch.
 *
 * We use an array larger than L3 cache so cache misses are real DRAM accesses.
 *
 * Bandwidth calculation:
 *   bandwidth (GB/s) = (bytes_accessed) / (elapsed_time_seconds * 1e9)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* 256 MB array — comfortably exceeds L3 cache on most machines */
#define ARRAY_BYTES    (256ULL * 1024 * 1024)
#define ARRAY_ELEMS    (ARRAY_BYTES / sizeof(uint64_t))

/*
 * Fisher-Yates shuffle to generate a random permutation of indices.
 * We'll follow this chain: next = indices[i] → indices[next] → ...
 * This creates a random-walk pointer chase that defeats prefetching.
 */
static void fisher_yates_shuffle(uint64_t *arr, uint64_t n) {
    /* Simple LCG random number generator (avoid stdlib rand quality issues) */
    uint64_t seed = 12345678901234567ULL;
    for (uint64_t i = n - 1; i > 0; i--) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        uint64_t j = seed % (i + 1);
        uint64_t tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

int main(void) {
    printf("Sequential vs Random Access Benchmark\n");
    printf("======================================\n");
    printf("Array size: %llu MB (%llu elements of uint64_t)\n\n",
           (unsigned long long)(ARRAY_BYTES / (1024*1024)),
           (unsigned long long)ARRAY_ELEMS);

    /* Allocate data array */
    uint64_t *data = malloc(ARRAY_BYTES);
    if (!data) { perror("malloc data"); return 1; }
    for (uint64_t i = 0; i < ARRAY_ELEMS; i++) data[i] = i;

    /* Allocate index array for random access */
    uint64_t *indices = malloc(ARRAY_ELEMS * sizeof(uint64_t));
    if (!indices) { perror("malloc indices"); return 1; }
    for (uint64_t i = 0; i < ARRAY_ELEMS; i++) indices[i] = i;
    fisher_yates_shuffle(indices, ARRAY_ELEMS);

    /* ---- SEQUENTIAL ACCESS ---- */
    {
        volatile uint64_t sink = 0;
        /* Warm up (touch every page) */
        for (uint64_t i = 0; i < ARRAY_ELEMS; i++) sink += data[i];

        uint64_t start = now_ns();
        sink = 0;
        for (uint64_t i = 0; i < ARRAY_ELEMS; i++) {
            sink += data[i];
        }
        uint64_t end = now_ns();

        if (sink == 0) printf("(sink)\n");

        double elapsed = (double)(end - start) / 1e9;
        double bw_gbs  = (double)ARRAY_BYTES / elapsed / 1e9;
        double ns_elem = (double)(end - start) / (double)ARRAY_ELEMS;

        printf("Sequential access:\n");
        printf("  Time:       %.3f s\n", elapsed);
        printf("  Bandwidth:  %.2f GB/s\n", bw_gbs);
        printf("  ns/element: %.2f\n\n", ns_elem);
    }

    /* ---- RANDOM ACCESS (pointer chase via shuffled indices) ---- */
    {
        volatile uint64_t sink = 0;
        /* No warm-up — random access defeats the prefetcher anyway */

        uint64_t start = now_ns();
        for (uint64_t i = 0; i < ARRAY_ELEMS; i++) {
            sink += data[indices[i]];
        }
        uint64_t end = now_ns();

        if (sink == 0) printf("(sink)\n");

        double elapsed = (double)(end - start) / 1e9;
        double bw_gbs  = (double)ARRAY_BYTES / elapsed / 1e9;
        double ns_elem = (double)(end - start) / (double)ARRAY_ELEMS;

        printf("Random access:\n");
        printf("  Time:       %.3f s\n", elapsed);
        printf("  Bandwidth:  %.2f GB/s\n", bw_gbs);
        printf("  ns/element: %.2f\n\n", ns_elem);
    }

    printf("Conclusion:\n");
    printf("  Sequential is fast because hardware prefetchers detect the\n");
    printf("  linear pattern and load cache lines before they are needed.\n");
    printf("  Random access stalls for DRAM every single access (~100 ns).\n");
    printf("  This is why data layout matters enormously in ML workloads.\n");

    free(data);
    free(indices);
    return 0;
}
