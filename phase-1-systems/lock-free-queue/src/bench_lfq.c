/*
 * bench_lfq.c — Lock-free queue vs mutex-protected queue benchmark.
 *
 * Both queues process the same workload:
 *   NUM_PRODUCERS producer threads × ITEMS_PER_PROD enqueues
 *   NUM_CONSUMERS consumer threads share the total work
 *
 * MUTEX QUEUE IMPLEMENTATION (in this file):
 *   A dynamically-growing ring buffer protected by a single pthread_mutex.
 *   This is representative of a naively-correct concurrent queue and is
 *   a fair comparison baseline.
 *
 * WHAT TO EXPECT (see BENCHMARKS.md for details):
 *   - Under high thread contention the lock-free queue typically wins
 *     because threads never block each other — they retry instead.
 *   - Under low contention a mutex queue can be faster because a single
 *     CAS is cheaper than the spinning + helping logic of the MS queue.
 *   - The difference narrows when the OS scheduler is cooperative.
 */

#include "lfq.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* -------------------------------------------------------------------------
 * Configuration — same as stress test
 * ---------------------------------------------------------------------- */

#define NUM_PRODUCERS   4
#define NUM_CONSUMERS   4
#define ITEMS_PER_PROD  250000
#define TOTAL_ITEMS     (NUM_PRODUCERS * ITEMS_PER_PROD)

/* -------------------------------------------------------------------------
 * Simple mutex-protected queue (linked-list, no capacity limit)
 * ---------------------------------------------------------------------- */

typedef struct MQNode {
    struct MQNode *next;
    void          *data;
} MQNode;

typedef struct {
    MQNode         *head;
    MQNode         *tail;
    pthread_mutex_t lock;
} MutexQueue;

static MutexQueue *mq_new(void)
{
    MutexQueue *q = malloc(sizeof *q);
    if (!q) abort();
    /* Use a sentinel just like the lock-free queue for a fair comparison. */
    MQNode *sentinel = malloc(sizeof *sentinel);
    if (!sentinel) abort();
    sentinel->next = NULL;
    sentinel->data = NULL;
    q->head = q->tail = sentinel;
    pthread_mutex_init(&q->lock, NULL);
    return q;
}

static void mq_enqueue(MutexQueue *q, void *data)
{
    MQNode *n = malloc(sizeof *n);
    if (!n) abort();
    n->next = NULL;
    n->data = data;

    pthread_mutex_lock(&q->lock);
    q->tail->next = n;
    q->tail       = n;
    pthread_mutex_unlock(&q->lock);
}

static int mq_dequeue(MutexQueue *q, void **out)
{
    pthread_mutex_lock(&q->lock);
    MQNode *head = q->head;
    MQNode *next = head->next;
    if (next == NULL) {
        pthread_mutex_unlock(&q->lock);
        return 0;
    }
    *out    = next->data;
    q->head = next;
    pthread_mutex_unlock(&q->lock);
    free(head);  /* free old sentinel */
    return 1;
}

static void mq_free(MutexQueue *q)
{
    if (!q) return;
    MQNode *cur = q->head;
    while (cur) {
        MQNode *nxt = cur->next;
        free(cur);
        cur = nxt;
    }
    pthread_mutex_destroy(&q->lock);
    free(q);
}

/* -------------------------------------------------------------------------
 * Shared benchmark state (one set per run, reset between lock-free / mutex)
 * ---------------------------------------------------------------------- */

typedef struct {
    /* Either a lock-free queue or a mutex queue pointer — only one is set */
    LFQueue    *lfq;
    MutexQueue *mq;

    atomic_int  consumed;
    atomic_int  produced;
} BenchState;

static BenchState g_state;

/* -------------------------------------------------------------------------
 * Thread functions — generic, switch on which queue type is set
 * ---------------------------------------------------------------------- */

static void *bench_producer(void *arg)
{
    (void)arg;
    for (int i = 0; i < ITEMS_PER_PROD; i++) {
        void *item = (void *)(intptr_t)(i + 1);
        if (g_state.lfq)
            lfq_enqueue(g_state.lfq, item);
        else
            mq_enqueue(g_state.mq, item);
        atomic_fetch_add_explicit(&g_state.produced, 1, memory_order_relaxed);
    }
    return NULL;
}

static void *bench_consumer(void *arg)
{
    (void)arg;
    void *item;
    while (atomic_load_explicit(&g_state.consumed, memory_order_relaxed) < TOTAL_ITEMS) {
        int ok;
        if (g_state.lfq)
            ok = lfq_dequeue(g_state.lfq, &item);
        else
            ok = mq_dequeue(g_state.mq, &item);
        if (ok)
            atomic_fetch_add_explicit(&g_state.consumed, 1, memory_order_relaxed);
    }
    return NULL;
}

/* -------------------------------------------------------------------------
 * Timing
 * ---------------------------------------------------------------------- */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* -------------------------------------------------------------------------
 * Run one benchmark trial, return elapsed seconds
 * ---------------------------------------------------------------------- */

static double run_trial(void)
{
    atomic_init(&g_state.consumed, 0);
    atomic_init(&g_state.produced, 0);

    pthread_t producers[NUM_PRODUCERS];
    pthread_t consumers[NUM_CONSUMERS];

    double t_start = now_sec();

    for (int i = 0; i < NUM_CONSUMERS; i++)
        pthread_create(&consumers[i], NULL, bench_consumer, NULL);
    for (int i = 0; i < NUM_PRODUCERS; i++)
        pthread_create(&producers[i], NULL, bench_producer, NULL);

    for (int i = 0; i < NUM_PRODUCERS; i++) pthread_join(producers[i], NULL);
    for (int i = 0; i < NUM_CONSUMERS; i++) pthread_join(consumers[i], NULL);

    return now_sec() - t_start;
}

/* -------------------------------------------------------------------------
 * Main
 * ---------------------------------------------------------------------- */

int main(void)
{
    printf("=== Lock-Free vs Mutex Queue Benchmark ===\n");
    printf("Producers: %d  |  Consumers: %d  |  Items/producer: %d\n\n",
           NUM_PRODUCERS, NUM_CONSUMERS, ITEMS_PER_PROD);

    /* ------------------------------------------------------------------ */
    /* Lock-free run                                                        */
    /* ------------------------------------------------------------------ */

    printf("Running lock-free queue...\n");
    g_state.lfq = lfq_new();
    g_state.mq  = NULL;
    double lf_time = run_trial();
    lfq_free(g_state.lfq);
    g_state.lfq = NULL;

    printf("  Lock-free time : %.4f s  (%.2f M ops/sec)\n",
           lf_time, (double)TOTAL_ITEMS / lf_time / 1e6);

    /* ------------------------------------------------------------------ */
    /* Mutex run                                                            */
    /* ------------------------------------------------------------------ */

    printf("Running mutex queue...\n");
    g_state.mq  = mq_new();
    g_state.lfq = NULL;
    double mx_time = run_trial();
    mq_free(g_state.mq);
    g_state.mq = NULL;

    printf("  Mutex time     : %.4f s  (%.2f M ops/sec)\n",
           mx_time, (double)TOTAL_ITEMS / mx_time / 1e6);

    /* ------------------------------------------------------------------ */
    /* Summary                                                              */
    /* ------------------------------------------------------------------ */

    double speedup = mx_time / lf_time;
    printf("\n--- Summary ---\n");
    printf("Lock-free : %.4f s\n", lf_time);
    printf("Mutex     : %.4f s\n", mx_time);

    if (speedup >= 1.0) {
        printf("Speedup   : %.2fx  (lock-free is faster)\n", speedup);
    } else {
        printf("Speedup   : %.2fx  (mutex is faster — low contention regime)\n",
               1.0 / speedup);
    }

    printf("\nInterpretation:\n");
    printf("  If lock-free wins: contention was high enough that avoiding\n");
    printf("  kernel-mode blocking (mutex sleep/wake) paid off.\n");
    printf("  If mutex wins: the CAS retry loop + cache-line ping-pong\n");
    printf("  cost more than the mutex overhead at this thread count.\n");

    return 0;
}
