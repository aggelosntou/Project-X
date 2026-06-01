/*
 * thread-pool/src/test_pool.c
 *
 * Three tests for the thread pool:
 *   1. Shared counter: 1000 tasks increment a mutex-protected counter.
 *      After tp_wait(), counter must equal 1000.
 *   2. Return values: each task writes its index into a result array.
 *      After tp_wait(), every slot must be filled correctly.
 *   3. Timing: compare single-thread baseline vs. pool for 100 CPU-bound
 *      tasks (busy-loop, not sleep) and report the speedup.
 */

#include "thread_pool.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* =========================================================================
 * Timing helper
 * ========================================================================= */

/* Return current time in seconds with nanosecond precision */
static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* =========================================================================
 * TEST 1: shared counter
 * ========================================================================= */

#define N_COUNTER_TASKS 1000

typedef struct {
    int            *counter;
    pthread_mutex_t *mutex;
} CounterArg;

static void increment_counter(void *arg)
{
    CounterArg *ca = (CounterArg *)arg;
    pthread_mutex_lock(ca->mutex);
    (*ca->counter)++;
    pthread_mutex_unlock(ca->mutex);
}

static void test_shared_counter(void)
{
    printf("Test 1: shared counter (%d tasks) ...\n", N_COUNTER_TASKS);

    int counter = 0;
    pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, NULL);

    /* All args point to the same counter/mutex; they differ only structurally.
     * Allocate an array so each task has its own CounterArg struct
     * (the struct contents are the same, but having separate structs avoids
     * any risk of one task reading a struct being written by tp_submit). */
    CounterArg args[N_COUNTER_TASKS];
    for (int i = 0; i < N_COUNTER_TASKS; i++) {
        args[i].counter = &counter;
        args[i].mutex   = &mutex;
    }

    ThreadPool *pool = tp_create(4);
    assert(pool != NULL);

    for (int i = 0; i < N_COUNTER_TASKS; i++) {
        int rc = tp_submit(pool, increment_counter, &args[i]);
        assert(rc == 0);
    }

    tp_wait(pool);

    assert(counter == N_COUNTER_TASKS);

    tp_destroy(pool);
    pthread_mutex_destroy(&mutex);

    printf("  counter = %d  (expected %d)  PASS\n", counter, N_COUNTER_TASKS);
}

/* =========================================================================
 * TEST 2: return values via result array
 * ========================================================================= */

#define N_RESULT_TASKS 200

typedef struct {
    int  index;
    int *result_array;
} ResultArg;

static void write_result(void *arg)
{
    ResultArg *ra = (ResultArg *)arg;
    /* Each task writes its own index into a dedicated slot.
     * No mutex needed: slots are independent (no two tasks share a slot). */
    ra->result_array[ra->index] = ra->index * ra->index;
}

static void test_return_values(void)
{
    printf("Test 2: result array (%d tasks) ...\n", N_RESULT_TASKS);

    int results[N_RESULT_TASKS];
    for (int i = 0; i < N_RESULT_TASKS; i++) results[i] = -1;  /* sentinel */

    ResultArg args[N_RESULT_TASKS];
    for (int i = 0; i < N_RESULT_TASKS; i++) {
        args[i].index        = i;
        args[i].result_array = results;
    }

    ThreadPool *pool = tp_create(8);
    assert(pool != NULL);

    for (int i = 0; i < N_RESULT_TASKS; i++) {
        int rc = tp_submit(pool, write_result, &args[i]);
        assert(rc == 0);
    }

    tp_wait(pool);

    /* Verify every slot was written correctly */
    for (int i = 0; i < N_RESULT_TASKS; i++) {
        assert(results[i] == i * i);
    }

    tp_destroy(pool);

    printf("  All %d results correct  PASS\n", N_RESULT_TASKS);
}

/* =========================================================================
 * TEST 3: timing — pool vs. single thread
 * ========================================================================= */

#define N_TIMING_TASKS 100

/* CPU-bound busy loop: count up to `iters` */
static void busy_task(void *arg)
{
    volatile long iters = *(long *)arg;
    volatile long x = 0;
    for (long i = 0; i < iters; i++) x += i;
    (void)x;
}

static void test_timing(void)
{
    printf("Test 3: timing — %d CPU-bound tasks ...\n", N_TIMING_TASKS);

    long iters = 20000000L;   /* 20 million iterations per task */

    long iter_args[N_TIMING_TASKS];
    for (int i = 0; i < N_TIMING_TASKS; i++) iter_args[i] = iters;

    /* --- single-thread baseline --- */
    double t0 = now_seconds();
    for (int i = 0; i < N_TIMING_TASKS; i++) {
        busy_task(&iter_args[i]);
    }
    double single_thread_time = now_seconds() - t0;

    /* --- thread pool --- */
    /* Use as many threads as we can reasonably assume (4).
     * On a machine with fewer cores, speedup will be <4. */
    int num_threads = 4;
    ThreadPool *pool = tp_create(num_threads);
    assert(pool != NULL);

    t0 = now_seconds();
    for (int i = 0; i < N_TIMING_TASKS; i++) {
        int rc = tp_submit(pool, busy_task, &iter_args[i]);
        assert(rc == 0);
    }
    tp_wait(pool);
    double pool_time = now_seconds() - t0;

    tp_destroy(pool);

    double speedup = single_thread_time / pool_time;

    printf("  Single-thread: %.3f s\n", single_thread_time);
    printf("  Pool (%d threads): %.3f s\n", num_threads, pool_time);
    printf("  Speedup: %.2fx\n", speedup);

    /* The pool should be faster (speedup > 1) on any real multi-core machine.
     * We use a very loose bound (1.5x) to avoid false failures on CI machines
     * with limited CPU resources. */
    if (speedup >= 1.5) {
        printf("  Speedup >= 1.5x  PASS\n");
    } else {
        printf("  Speedup < 1.5x  (OK on single-core or loaded machine)\n");
    }
}

/* =========================================================================
 * main
 * ========================================================================= */

int main(void)
{
    printf("Running thread-pool tests...\n\n");

    test_shared_counter();
    printf("\n");
    test_return_values();
    printf("\n");
    test_timing();

    printf("\nAll tests passed.\n");
    return 0;
}
