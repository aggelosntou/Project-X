#ifndef THREAD_POOL_H
#define THREAD_POOL_H
#include <stddef.h>

typedef void (*task_fn)(void *arg);

typedef struct ThreadPool ThreadPool;

ThreadPool *tp_create(int num_threads);
int         tp_submit(ThreadPool *pool, task_fn fn, void *arg);
void        tp_wait(ThreadPool *pool);
void        tp_destroy(ThreadPool *pool);

#endif
