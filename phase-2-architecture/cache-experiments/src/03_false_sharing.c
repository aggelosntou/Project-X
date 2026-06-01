/*
 * 03_false_sharing.c — Demonstrate the false sharing problem
 *
 * FALSE SHARING: Two threads each modify their "own" variable, but those
 * variables happen to live on the SAME cache line (64 bytes). The CPU cache
 * coherence protocol (MESI) forces both cores to invalidate and reload the
 * cache line on every write by the other core. This causes massive slowdown
 * even though the threads are technically touching different variables.
 *
 * Fix: pad the variables to be on SEPARATE cache lines.
 *
 * Expected output:
 *   Adjacent (false sharing): ~800ms
 *   Padded (no false sharing): ~150ms
 *   Speedup: ~5x
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>

#define ITERATIONS  200000000ULL   /* 200 million increments per thread */
#define CACHE_LINE  64

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ---- VERSION 1: Adjacent counters (false sharing) ---- */
typedef struct {
    volatile uint64_t counter0;   /* Thread 0 uses this */
    volatile uint64_t counter1;   /* Thread 1 uses this */
    /* These two are adjacent → SAME cache line (128 bits = 16 bytes << 64) */
} AdjacentCounters;

/* ---- VERSION 2: Padded counters (no false sharing) ---- */
typedef struct {
    volatile uint64_t counter;
    /* Pad to fill a complete cache line. sizeof(uint64_t) = 8,
     * so we need 64 - 8 = 56 bytes of padding.               */
    char _pad[CACHE_LINE - sizeof(uint64_t)];
} PaddedCounter;

typedef struct {
    void            *counters_base;
    int              index;          /* 0 or 1 */
    int              mode;           /* 0 = adjacent, 1 = padded */
    uint64_t         iterations;
} ThreadArgs;

static void *worker_adjacent(void *arg) {
    ThreadArgs      *a = (ThreadArgs *)arg;
    AdjacentCounters *c = (AdjacentCounters *)a->counters_base;

    if (a->index == 0) {
        for (uint64_t i = 0; i < a->iterations; i++) c->counter0++;
    } else {
        for (uint64_t i = 0; i < a->iterations; i++) c->counter1++;
    }
    return NULL;
}

static void *worker_padded(void *arg) {
    ThreadArgs   *a = (ThreadArgs *)arg;
    PaddedCounter *c = (PaddedCounter *)a->counters_base;

    for (uint64_t i = 0; i < a->iterations; i++) c[a->index].counter++;
    return NULL;
}

static double run_adjacent(uint64_t iterations) {
    AdjacentCounters counters = {0, 0};

    ThreadArgs args[2];
    pthread_t  threads[2];

    for (int i = 0; i < 2; i++) {
        args[i].counters_base = &counters;
        args[i].index         = i;
        args[i].iterations    = iterations;
    }

    uint64_t start = now_ns();
    for (int i = 0; i < 2; i++) pthread_create(&threads[i], NULL, worker_adjacent, &args[i]);
    for (int i = 0; i < 2; i++) pthread_join(threads[i], NULL);
    uint64_t end = now_ns();

    printf("  Adjacent counters: c0=%llu, c1=%llu\n",
           (unsigned long long)counters.counter0,
           (unsigned long long)counters.counter1);

    return (double)(end - start) / 1e9;
}

static double run_padded(uint64_t iterations) {
    /* Align to cache line boundary */
    PaddedCounter *counters = aligned_alloc(CACHE_LINE, 2 * sizeof(PaddedCounter));
    if (!counters) { perror("aligned_alloc"); exit(1); }
    counters[0].counter = 0;
    counters[1].counter = 0;

    ThreadArgs args[2];
    pthread_t  threads[2];

    for (int i = 0; i < 2; i++) {
        args[i].counters_base = counters;
        args[i].index         = i;
        args[i].iterations    = iterations;
    }

    uint64_t start = now_ns();
    for (int i = 0; i < 2; i++) pthread_create(&threads[i], NULL, worker_padded, &args[i]);
    for (int i = 0; i < 2; i++) pthread_join(threads[i], NULL);
    uint64_t end = now_ns();

    printf("  Padded counters:   c0=%llu, c1=%llu\n",
           (unsigned long long)counters[0].counter,
           (unsigned long long)counters[1].counter);

    free(counters);
    return (double)(end - start) / 1e9;
}

int main(void) {
    printf("False Sharing Demonstration\n");
    printf("===========================\n");
    printf("Each thread increments its own counter %llu times.\n\n",
           (unsigned long long)ITERATIONS);

    printf("sizeof(uint64_t)    = %zu bytes\n", sizeof(uint64_t));
    printf("sizeof(PaddedCounter) = %zu bytes (= cache line)\n\n", sizeof(PaddedCounter));

    /* Verify padding is exactly one cache line */
    if (sizeof(PaddedCounter) != CACHE_LINE) {
        fprintf(stderr, "Warning: PaddedCounter size %zu != cache line %d\n",
                sizeof(PaddedCounter), CACHE_LINE);
    }

    printf("[1] Adjacent counters (false sharing)...\n");
    double t_adjacent = run_adjacent(ITERATIONS);

    printf("[2] Padded counters (no false sharing)...\n");
    double t_padded = run_padded(ITERATIONS);

    printf("\nResults:\n");
    printf("  Adjacent (false sharing): %.3f s\n", t_adjacent);
    printf("  Padded (no false sharing): %.3f s\n",  t_padded);
    printf("  Speedup: %.1fx\n", t_adjacent / t_padded);

    printf("\nConclusion:\n");
    printf("  Even though each thread touches its own variable, the CPU's\n");
    printf("  cache coherence protocol must broadcast every write to the other\n");
    printf("  core, because they share a cache line. Padding forces each\n");
    printf("  variable into its own line, eliminating this overhead.\n");

    return 0;
}
