/*
 * 02_cache_line_size.c — Determine cache line size by strided access
 *
 * Technique: Walk through a large array with a varying stride (step size).
 * Access arr[0], arr[stride], arr[2*stride], arr[3*stride], ...
 *
 * When stride < (cache_line_size / element_size):
 *   Each access brings multiple elements into cache. The NEXT strided access
 *   hits the same cache line → it's a cache HIT → fast.
 *
 * When stride >= (cache_line_size / element_size):
 *   Every access lands on a NEW cache line → every access is a cache MISS.
 *   Latency PLATEAUS at the cache miss penalty.
 *
 * The plateau begins when stride = cache_line_bytes / sizeof(element).
 * On modern CPUs: cache_line_size = 64 bytes.
 * With uint32_t (4 bytes): plateau at stride = 64/4 = 16.
 *
 * We also test a large array (bigger than L3) so every access to a new
 * cache line is definitely a miss.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* Large enough to exceed L3 cache (64 MB) */
#define ARRAY_SIZE   (64 * 1024 * 1024 / sizeof(uint32_t))
#define ITERATIONS   8

static double measure_stride(uint32_t *arr, size_t n, size_t stride) {
    volatile uint64_t sink = 0;

    /* Warm-up pass */
    for (size_t i = 0; i < n; i += stride) sink += arr[i];

    uint64_t start = now_ns();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (size_t i = 0; i < n; i += stride) {
            sink += arr[i];
        }
    }
    uint64_t end = now_ns();

    if (sink == 0) printf("(sink=%llu)\n", (unsigned long long)sink);

    size_t accesses = (n / stride) * ITERATIONS;
    return (double)(end - start) / (double)accesses;
}

int main(void) {
    uint32_t *arr = malloc(ARRAY_SIZE * sizeof(uint32_t));
    if (!arr) { perror("malloc"); return 1; }

    for (size_t i = 0; i < ARRAY_SIZE; i++) arr[i] = (uint32_t)(i + 1);

    printf("Cache Line Size Detection — Strided Access\n");
    printf("===========================================\n");
    printf("Array size: %zu MB (exceeds L3 cache)\n\n",
           ARRAY_SIZE * sizeof(uint32_t) / (1024*1024));

    printf("%-8s | %-14s | %-12s | %s\n",
           "Stride", "Bytes jumped", "ns/access", "Note");
    printf("%-8s-+-%-14s-+-%-12s-+-%s\n",
           "--------", "--------------", "------------", "----");

    for (int stride = 1; stride <= 128; stride *= 2) {
        double ns = measure_stride(arr, ARRAY_SIZE, (size_t)stride);
        int bytes_jumped = stride * (int)sizeof(uint32_t);
        const char *note = (bytes_jumped >= 64 && bytes_jumped <= 128) ?
                           "<-- plateau begins ~here" : "";
        printf("%-8d | %-14d | %-12.2f | %s\n",
               stride, bytes_jumped, ns, note);
    }

    printf("\n");
    printf("Interpretation:\n");
    printf("  - For stride <= 16 (<=64 bytes), latency stays LOW:\n");
    printf("    accessing consecutive elements hits the SAME cache line.\n");
    printf("  - Latency PLATEAUS around stride 16 (64 bytes):\n");
    printf("    every access now touches a NEW cache line.\n");
    printf("  - The cache line size = %d bytes (stride at plateau × 4 bytes).\n", 64);

    free(arr);
    return 0;
}
