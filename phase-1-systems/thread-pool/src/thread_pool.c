/*
 * thread-pool/src/thread_pool.c
 *
 * A fixed-size thread pool backed by a circular buffer task queue.
 *
 * DESIGN OVERVIEW
 * ---------------
 *
 *  tp_create()
 *    ├── allocates the ThreadPool struct
 *    ├── initialises mutex, not_empty cond var, all_done cond var
 *    └── spawns num_threads worker threads each running worker_loop()
 *
 *  tp_submit()
 *    └── locks mutex → pushes task into circular buffer → increments pending
 *        → signals not_empty → unlocks
 *
 *  tp_wait()
 *    └── locks mutex → while pending > 0, cond_wait(all_done) → unlocks
 *
 *  tp_destroy()
 *    └── locks mutex → sets shutdown=1 → broadcasts not_empty → unlocks
 *        → joins all threads → frees memory
 *
 *  worker_loop() (runs in each thread)
 *    └── loop:
 *          lock mutex
 *          while queue is empty AND !shutdown → cond_wait(not_empty)
 *          if shutdown AND queue is empty → unlock and exit
 *          pop task from queue
 *          unlock mutex
 *          execute task (outside mutex — this is the key for parallelism)
 *          lock mutex
 *          decrement pending
 *          if pending == 0 → signal all_done
 *          unlock mutex
 *
 * WHY condition variables need a while loop (spurious wakeups):
 *   The POSIX spec explicitly allows cond_wait to return even when no signal
 *   was sent — so-called "spurious wakeups", caused by signal delivery or OS
 *   scheduler quirks. If we used `if` instead of `while`, the thread would
 *   proceed even if the queue is still empty, leading to out-of-bounds reads.
 *   The while loop re-checks the predicate every time we wake up.
 *
 * WHY pending needs the mutex:
 *   `pending` is a shared integer modified by both the submitting thread
 *   (tp_submit increments it) and the worker threads (decremented after
 *   each task). Without a mutex, concurrent increment/decrement is a data
 *   race — undefined behaviour in C11. The mutex also synchronises with
 *   cond_wait(all_done): the signal is only meaningful if tp_wait checks
 *   `pending` atomically with the signal (both protected by the same mutex).
 *
 * WHAT HAPPENS if you destroy the pool with tasks in flight:
 *   tp_destroy sets shutdown=1 and broadcasts. Workers drain the remaining
 *   queue before exiting (they exit only when BOTH shutdown==1 AND the queue
 *   is empty). So in-queue tasks complete. However, tasks that were popped
 *   and are actively executing will finish because we join all threads —
 *   join blocks until the thread's function returns. No task is abandoned.
 *   IMPORTANT: after tp_destroy returns, the pool pointer is invalid. Any
 *   tp_submit call after that is undefined behaviour.
 */

#include "thread_pool.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Maximum number of tasks that can be queued before tp_submit blocks.
 * A real implementation might grow the queue dynamically. */
#define QUEUE_SIZE 4096

/* -------------------------------------------------------------------------
 * Internal types
 * ------------------------------------------------------------------------- */

typedef struct {
    task_fn  fn;
    void    *arg;
} Task;

struct ThreadPool {
    /* Worker threads */
    pthread_t *threads;
    int        num_threads;

    /* Circular buffer task queue */
    Task   queue[QUEUE_SIZE];
    int    head;   /* index of next task to pop  */
    int    tail;   /* index where next task goes */
    int    count;  /* number of tasks in queue   */

    /* Synchronisation */
    pthread_mutex_t mutex;
    pthread_cond_t  not_empty;  /* signalled when a task is pushed    */
    pthread_cond_t  all_done;   /* signalled when pending drops to 0  */

    /* Accounting */
    int pending;   /* tasks submitted but not yet completed */
    int shutdown;  /* set to 1 by tp_destroy                */
};

/* -------------------------------------------------------------------------
 * Worker thread function
 * ------------------------------------------------------------------------- */

static void *worker_loop(void *arg)
{
    ThreadPool *pool = (ThreadPool *)arg;

    for (;;) {
        pthread_mutex_lock(&pool->mutex);

        /* Wait until there is work to do or we are shutting down.
         * MUST be a while loop — see design note on spurious wakeups above. */
        while (pool->count == 0 && !pool->shutdown) {
            pthread_cond_wait(&pool->not_empty, &pool->mutex);
        }

        /* If shutdown and nothing left in queue, this thread is done. */
        if (pool->shutdown && pool->count == 0) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }

        /* Pop the next task from the circular buffer */
        Task task = pool->queue[pool->head];
        pool->head  = (pool->head + 1) % QUEUE_SIZE;
        pool->count--;

        pthread_mutex_unlock(&pool->mutex);

        /* ----------------------------------------------------------------
         * Execute OUTSIDE the mutex.
         * This is the whole point of a thread pool: tasks run in parallel.
         * If we held the mutex here, every task would run sequentially —
         * no better than a single-threaded loop.
         * ---------------------------------------------------------------- */
        task.fn(task.arg);

        /* Mark this task as completed */
        pthread_mutex_lock(&pool->mutex);
        pool->pending--;
        if (pool->pending == 0) {
            /* Wake up anyone blocked in tp_wait() */
            pthread_cond_signal(&pool->all_done);
        }
        pthread_mutex_unlock(&pool->mutex);
    }

    return NULL;
}

/* -------------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------------- */

ThreadPool *tp_create(int num_threads)
{
    if (num_threads <= 0) return NULL;

    ThreadPool *pool = calloc(1, sizeof(ThreadPool));
    if (pool == NULL) return NULL;

    pool->num_threads = num_threads;
    pool->head        = 0;
    pool->tail        = 0;
    pool->count       = 0;
    pool->pending     = 0;
    pool->shutdown    = 0;

    pthread_mutex_init(&pool->mutex,     NULL);
    pthread_cond_init (&pool->not_empty, NULL);
    pthread_cond_init (&pool->all_done,  NULL);

    pool->threads = malloc((size_t)num_threads * sizeof(pthread_t));
    if (pool->threads == NULL) {
        free(pool);
        return NULL;
    }

    for (int i = 0; i < num_threads; i++) {
        int rc = pthread_create(&pool->threads[i], NULL, worker_loop, pool);
        if (rc != 0) {
            /* Partial creation: shut down threads that did start */
            pool->shutdown = 1;
            pthread_cond_broadcast(&pool->not_empty);
            for (int j = 0; j < i; j++) pthread_join(pool->threads[j], NULL);
            pthread_mutex_destroy(&pool->mutex);
            pthread_cond_destroy(&pool->not_empty);
            pthread_cond_destroy(&pool->all_done);
            free(pool->threads);
            free(pool);
            return NULL;
        }
    }

    return pool;
}

int tp_submit(ThreadPool *pool, task_fn fn, void *arg)
{
    if (pool == NULL || fn == NULL) return -1;

    pthread_mutex_lock(&pool->mutex);

    if (pool->shutdown) {
        pthread_mutex_unlock(&pool->mutex);
        return -1;
    }

    /* Block if the queue is full (simple back-pressure) */
    while (pool->count == QUEUE_SIZE) {
        /* In a more sophisticated design we'd grow the queue instead.
         * Here we wait for a slot to free up. */
        pthread_cond_wait(&pool->not_empty, &pool->mutex);
    }

    /* Push task into the circular buffer */
    pool->queue[pool->tail].fn  = fn;
    pool->queue[pool->tail].arg = arg;
    pool->tail    = (pool->tail + 1) % QUEUE_SIZE;
    pool->count++;
    pool->pending++;

    /* Wake at least one worker */
    pthread_cond_signal(&pool->not_empty);

    pthread_mutex_unlock(&pool->mutex);
    return 0;
}

void tp_wait(ThreadPool *pool)
{
    if (pool == NULL) return;

    pthread_mutex_lock(&pool->mutex);

    /* Wait until all submitted tasks have completed.
     * While loop guards against spurious wakeups. */
    while (pool->pending > 0) {
        pthread_cond_wait(&pool->all_done, &pool->mutex);
    }

    pthread_mutex_unlock(&pool->mutex);
}

void tp_destroy(ThreadPool *pool)
{
    if (pool == NULL) return;

    /* Signal shutdown to all workers */
    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->not_empty);  /* wake ALL sleeping workers */
    pthread_mutex_unlock(&pool->mutex);

    /* Wait for every worker to finish its current task and exit */
    for (int i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }

    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->not_empty);
    pthread_cond_destroy(&pool->all_done);

    free(pool->threads);
    free(pool);
}
