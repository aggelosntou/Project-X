# LEARNINGS — thread-pool

## 1. pthreads API

### `pthread_create`

```c
int pthread_create(pthread_t *thread,
                   const pthread_attr_t *attr,
                   void *(*start_routine)(void *),
                   void *arg);
```

Spawns a new OS thread that calls `start_routine(arg)`. Returns 0 on success, error code on failure. The `pthread_t` is an opaque handle used for `pthread_join`.

### `pthread_join`

```c
int pthread_join(pthread_t thread, void **retval);
```

Blocks the calling thread until `thread` exits. Essential in `tp_destroy` to avoid returning while workers are still running — without join, the workers would reference freed pool memory after `free(pool)`.

### `pthread_mutex_t`

A mutual exclusion lock. Only one thread may hold it at a time.

```c
pthread_mutex_init(&mutex, NULL);   // initialise (or use PTHREAD_MUTEX_INITIALIZER)
pthread_mutex_lock(&mutex);         // acquire (blocks if already held)
pthread_mutex_unlock(&mutex);       // release
pthread_mutex_destroy(&mutex);      // free resources
```

### `pthread_cond_t`

A condition variable. Used to efficiently wait for a condition to become true.

```c
pthread_cond_init(&cond, NULL);
pthread_cond_wait(&cond, &mutex);   // atomically unlock mutex, sleep; re-lock on wake
pthread_cond_signal(&cond);         // wake one waiting thread
pthread_cond_broadcast(&cond);      // wake ALL waiting threads
pthread_cond_destroy(&cond);
```

`pthread_cond_wait` requires the associated mutex to be locked when called. It atomically releases the mutex and suspends the thread. When a signal arrives, it atomically reacquires the mutex before returning. This atomicity is what prevents missed signals.

---

## 2. Why condition variables need a `while` loop (spurious wakeups)

The POSIX standard explicitly permits `pthread_cond_wait` to return without any corresponding `pthread_cond_signal` or `pthread_cond_broadcast`. This is called a **spurious wakeup** and is caused by:

- Signal delivery to the process interrupting the underlying `futex` system call.
- Scheduler optimisations that requeue waiting threads.
- Some hardware platforms where `cond_wait` is implemented differently.

**Wrong (using `if`)**:

```c
// Thread wakes up even if count is still 0 → pops garbage from queue
if (pool->count == 0 && !pool->shutdown) {
    pthread_cond_wait(&pool->not_empty, &pool->mutex);
}
// BUG: count may still be 0 here
pop_task(pool);
```

**Correct (using `while`)**:

```c
// Re-check the predicate every time we wake up
while (pool->count == 0 && !pool->shutdown) {
    pthread_cond_wait(&pool->not_empty, &pool->mutex);
}
// SAFE: if we exit this loop, either count > 0 or shutdown == 1
```

The `while` loop also correctly handles the case where two threads are woken by one signal (another valid POSIX behaviour) and only one of them finds work.

---

## 3. Why `pending` needs the mutex

`pending` is an `int` shared between:
- The submitting thread(s) (incremented in `tp_submit`)
- All worker threads (decremented after each task completes)

In C11, reading and writing a plain `int` from multiple threads without synchronisation is a **data race**, which is undefined behaviour. The CPU may reorder reads/writes; the compiler may cache the value in a register; a half-updated value may be observed.

Beyond correctness: `tp_wait` uses `pending` as the predicate for `cond_wait(all_done)`. For condition variables to work correctly, the predicate and the cond_wait must be protected by the **same mutex**. If `pending` were updated outside the mutex:

```
tp_wait:                              worker:
  lock mutex
  check pending == 0? → no (it's 1)
                                        -- pending becomes 0 here (outside mutex)
                                        signal all_done (but nobody is waiting yet!)
  cond_wait(all_done, mutex)  ← HANGS FOREVER: signal already fired
```

With the mutex protecting `pending` in the worker, the signal is always sent while `tp_wait` is either still holding the mutex (and hasn't called `cond_wait` yet) or is already sleeping in `cond_wait`. `cond_wait` releases the mutex atomically before sleeping, so the signal cannot be lost.

---

## 4. What happens if you destroy the pool with tasks in flight

**"In flight"** has two cases:

### Case A: tasks still in the queue (not yet started)

`tp_destroy` sets `shutdown = 1` and broadcasts `not_empty`. Workers wake up, check `shutdown && count == 0`. But if the queue still has tasks (`count > 0`), they **drain the queue first** — the worker loop only exits when both `shutdown == 1` AND `count == 0`. So queued tasks complete.

### Case B: tasks currently executing

A worker executing a task has already popped it from the queue and released the mutex. `tp_destroy` calls `pthread_join` on all threads, which blocks until each thread's `worker_loop` function returns. A thread executing a task cannot return from `worker_loop` until the task finishes and then the thread re-checks the loop condition. Therefore `pthread_join` blocks until all executing tasks have completed.

**In summary**: after `tp_destroy` returns, every task that was ever submitted (queued or running) has completed. No task is abandoned or lost.

**What you must not do**: call `tp_submit` after `tp_destroy`. The pool memory has been freed; writing to it is undefined behaviour. The implementation guards against this by checking `pool->shutdown` at the start of `tp_submit` and returning -1 immediately.
