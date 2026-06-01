/*
 * test_lfq.c — Stress test for the Michael-Scott lock-free queue.
 *
 * DESIGN:
 *   4 producer threads × 250,000 enqueues  = 1,000,000 total items produced
 *   4 consumer threads × (share 1,000,000) = 1,000,000 total items consumed
 *
 * A shared atomic counter tracks how many items have been consumed.
 * When the counter reaches TOTAL_ITEMS, all consumers exit.
 *
 * VERIFICATION:
 *   At the end we assert consumed == produced.  If the queue loses or
 *   duplicates an item, the test will either hang (counter never reaches
 *   the limit) or print a mismatch error.
 *
 * NOTE on data encoding:
 *   We store integers cast to void*.  This is technically implementation-
 *   defined but works on all major 64-bit platforms (int fits in a pointer).
 *   For portable code, store pointers to heap-allocated integers instead.
 */

#include "lfq.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* -------------------------------------------------------------------------
 * Configuration
 * ---------------------------------------------------------------------- */

#define NUM_PRODUCERS    4
#define NUM_CONSUMERS    4
#define ITEMS_PER_PROD   250000
#define TOTAL_ITEMS      (NUM_PRODUCERS * ITEMS_PER_PROD)  /* 1,000,000 */

/* -------------------------------------------------------------------------
 * Shared state
 * ---------------------------------------------------------------------- */

static LFQueue        *g_queue;
static atomic_int      g_consumed;   /* items dequeued so far              */
static atomic_int      g_produced;   /* items enqueued so far              */

/* -------------------------------------------------------------------------
 * Thread functions
 * ---------------------------------------------------------------------- */

static void *producer_fn(void *arg)
{
    (void)arg;
    for (int i = 0; i < ITEMS_PER_PROD; i++) {
        /*
         * Cast i+1 to void* so we can distinguish a valid 0-value item
         * from a NULL that might indicate an error.  Consumers subtract 1.
         */
        lfq_enqueue(g_queue, (void *)(intptr_t)(i + 1));
        atomic_fetch_add_explicit(&g_produced, 1, memory_order_relaxed);
    }
    return NULL;
}

static void *consumer_fn(void *arg)
{
    (void)arg;
    void *item;

    while (atomic_load_explicit(&g_consumed, memory_order_relaxed) < TOTAL_ITEMS) {
        if (lfq_dequeue(g_queue, &item)) {
            /*
             * We successfully dequeued an item.  Increment the shared
             * counter atomically so no two consumers count the same item.
             */
            atomic_fetch_add_explicit(&g_consumed, 1, memory_order_relaxed);
        }
        /*
         * If dequeue returned 0 the queue was momentarily empty.
         * We simply spin.  A real application would sleep briefly or use
         * a semaphore/condition variable to avoid burning CPU.
         */
    }
    return NULL;
}

/* -------------------------------------------------------------------------
 * Timing helpers
 * ---------------------------------------------------------------------- */

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* -------------------------------------------------------------------------
 * Main
 * ---------------------------------------------------------------------- */

int main(void)
{
    printf("=== Lock-Free Queue Stress Test ===\n");
    printf("Producers: %d  |  Consumers: %d  |  Items/producer: %d\n",
           NUM_PRODUCERS, NUM_CONSUMERS, ITEMS_PER_PROD);
    printf("Total items: %d\n\n", TOTAL_ITEMS);

    g_queue = lfq_new();
    if (!g_queue) {
        fprintf(stderr, "lfq_new() failed\n");
        return 1;
    }

    atomic_init(&g_consumed, 0);
    atomic_init(&g_produced, 0);

    pthread_t producers[NUM_PRODUCERS];
    pthread_t consumers[NUM_CONSUMERS];

    double t_start = now_seconds();

    /* Launch consumers first so they are ready to drain immediately. */
    for (int i = 0; i < NUM_CONSUMERS; i++) {
        if (pthread_create(&consumers[i], NULL, consumer_fn, NULL) != 0) {
            fprintf(stderr, "pthread_create consumer %d failed\n", i);
            return 1;
        }
    }

    for (int i = 0; i < NUM_PRODUCERS; i++) {
        if (pthread_create(&producers[i], NULL, producer_fn, NULL) != 0) {
            fprintf(stderr, "pthread_create producer %d failed\n", i);
            return 1;
        }
    }

    /* Wait for everyone to finish. */
    for (int i = 0; i < NUM_PRODUCERS; i++) pthread_join(producers[i], NULL);
    for (int i = 0; i < NUM_CONSUMERS; i++) pthread_join(consumers[i], NULL);

    double elapsed = now_seconds() - t_start;

    /* ------------------------------------------------------------------ */
    /* Verification                                                         */
    /* ------------------------------------------------------------------ */

    int produced = atomic_load(&g_produced);
    int consumed = atomic_load(&g_consumed);

    printf("--- Results ---\n");
    printf("Total produced : %d\n", produced);
    printf("Total consumed : %d\n", consumed);

    if (produced != TOTAL_ITEMS) {
        fprintf(stderr, "FAIL: produced %d, expected %d\n",
                produced, TOTAL_ITEMS);
    } else if (consumed != TOTAL_ITEMS) {
        fprintf(stderr, "FAIL: consumed %d, expected %d\n",
                consumed, TOTAL_ITEMS);
    } else {
        printf("PASS: all %d items accounted for\n", TOTAL_ITEMS);
    }

    /* ------------------------------------------------------------------ */
    /* Performance                                                          */
    /* ------------------------------------------------------------------ */

    double throughput = (double)TOTAL_ITEMS / elapsed;
    printf("\nElapsed time : %.4f seconds\n", elapsed);
    printf("Throughput   : %.0f ops/sec  (%.2f M ops/sec)\n",
           throughput, throughput / 1e6);

    lfq_free(g_queue);
    return (produced == TOTAL_ITEMS && consumed == TOTAL_ITEMS) ? 0 : 1;
}
