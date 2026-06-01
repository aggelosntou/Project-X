# thread-pool

A fixed-size thread pool in C using POSIX threads (`pthreads`). Worker threads wait on a condition variable and execute tasks from a shared circular buffer queue.

## API

```c
ThreadPool *tp_create(int num_threads);        // create pool with N workers
int         tp_submit(ThreadPool *, task_fn, void *arg); // enqueue a task
void        tp_wait(ThreadPool *);             // block until all tasks done
void        tp_destroy(ThreadPool *);          // drain, join, free
```

A `task_fn` is any `void (*)(void *)` function.

## Build

```bash
make
```

## Run

```bash
make run
```

Expected output:

```
Running thread-pool tests...

Test 1: shared counter (1000 tasks) ...
  counter = 1000  (expected 1000)  PASS

Test 2: result array (200 tasks) ...
  All 200 results correct  PASS

Test 3: timing — 100 CPU-bound tasks ...
  Single-thread: 1.234 s
  Pool (4 threads): 0.321 s
  Speedup: 3.84x
  Speedup >= 1.5x  PASS

All tests passed.
```

Actual timing and speedup depend on the number of CPU cores.

## Clean

```bash
make clean
```

## Internal design

```
submit → [mutex] → circular buffer → signal not_empty
                                              ↓
worker: wait(not_empty) → pop → unlock → execute → lock → --pending
                                                           ↓ pending==0?
                                                    signal all_done
                                                           ↓
tp_wait: wait(all_done) ← [mutex] ←───────────────────────┘
```

Key invariant: tasks execute **outside** the mutex so all worker threads run in parallel.

## File layout

```
src/
  thread_pool.h    — public API
  thread_pool.c    — implementation (pthreads, circular buffer)
  test_pool.c      — three tests (counter, results, timing)
Makefile
README.md
LEARNINGS.md
```
