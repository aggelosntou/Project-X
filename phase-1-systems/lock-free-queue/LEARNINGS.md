# Learnings — Lock-Free Queue

## 1. Compare-and-Swap (CAS)

CAS is the foundational atomic primitive for lock-free programming.

```
bool CAS(T *addr, T expected, T desired):
    if *addr == expected:
        *addr = desired
        return true   // "I won the race"
    else:
        return false  // "something changed; retry"
```

The CPU executes this as a single indivisible instruction (`LOCK CMPXCHG`
on x86, `LDAXR/STLXR` on ARM).  No other thread can observe a
half-completed CAS.

### Why CAS enables lock-free code

A mutex works by making a thread **sleep** until the lock is free.  With
CAS, a thread that loses the race simply **retries** — it never blocks.
This means:
- A thread that is preempted (descheduled) mid-operation cannot hold up
  other threads.
- Progress is guaranteed at the **system level** (at least one thread
  always makes forward progress), even if no individual thread is
  guaranteed to ever finish (that stronger property is called *wait-free*).

---

## 2. The ABA Problem

### Setup

Suppose two threads share a pointer `P` that currently holds value `A`.

```
Thread T1                        Thread T2
---------                        ---------
reads P → A
                                 removes A from the structure
                                 allocates a new node — allocator
                                   returns the *same address* A
                                 stores A back into P
T1's CAS(&P, A, next) succeeds!
```

T1 sees `P == A`, its CAS succeeds, but the *contents* of node A have
changed.  T1 is now operating on stale data while believing it is correct.

### Concrete example with a stack (simpler than a queue)

```
Stack: [A] → [B] → NULL          head = A
T1: reads head = A, next = B
T2: pops A, pops B, pushes A back   → stack: [A] → NULL
T1: CAS(&head, A, B) succeeds
    head is now B — but B was already freed!  Use-after-free.
```

### Why the sentinel approach partially mitigates ABA here

In this queue the **sentinel node is never freed** while the queue is
alive.  The `head` pointer only ever points at the sentinel.  Since the
sentinel's address never becomes available for reuse, a thread holding an
old value of `head` (pointing at the sentinel) cannot be confused by ABA
on the head pointer.

However, ABA can still occur on `q->tail` in theory.  Production-grade
solutions include:

| Technique | Idea |
|---|---|
| **Tagged pointers** | Pack a version counter into unused low/high bits; CAS on (pointer, counter) pair |
| **Hazard pointers** | Before accessing a node, publish its address; other threads defer freeing until no hazard pointer references it |
| **Epoch-based reclamation** | Divide time into epochs; defer frees until all threads have passed the epoch |

---

## 3. Memory Ordering

Modern CPUs and compilers reorder memory operations for performance.
`std::atomic` / `_Atomic` operations let you attach **ordering constraints**
to prevent problematic reorderings.

### The three orders used in this project

#### `memory_order_acquire` (on a load)

> "I will see all stores that the other thread had done **before** it
>  performed the release store that published this value."

Use this when reading a pointer that was published by another thread.
It ensures the pointed-to data is fully visible, not partially written.

```c
// Producer (release store):
atomic_store_explicit(&shared_ptr, new_node, memory_order_release);

// Consumer (acquire load):
LFQNode *p = atomic_load_explicit(&shared_ptr, memory_order_acquire);
// p->data is now guaranteed to be the value the producer wrote
```

#### `memory_order_release` (on a store)

> "All my prior stores must be visible to any thread that acquires
>  this value."

#### `memory_order_acq_rel` (on a read-modify-write like CAS)

> Both acquire and release at once.

A CAS is simultaneously a load (reading the old value) and a store
(writing the new value), so it needs both:
- **Release**: the data written before the CAS is visible after the CAS.
- **Acquire**: on failure, we re-read with acquire semantics so the
  retry loop sees the current state.

### What goes wrong without ordering

```c
// Incorrect (relaxed everywhere):
new_node->data = 42;
atomic_store_explicit(&q->tail->next, new_node, memory_order_relaxed);
// A consumer may observe tail->next = new_node but see data = garbage
// because the store to data was not ordered before the pointer store.
```

---

## 4. Lock-Free ≠ Wait-Free

| Property | Guarantee |
|---|---|
| **Obstruction-free** | A thread makes progress if all other threads pause |
| **Lock-free** | At least one thread makes progress at any time |
| **Wait-free** | Every thread makes progress in bounded steps |

The Michael-Scott queue is **lock-free**.  A single thread can spin
indefinitely (if it loses every CAS race), but the *system* always
makes progress.

Wait-free algorithms exist (e.g. Kogan-Petrank queue) but are more complex
and often slower in practice due to extra bookkeeping.

---

## 5. When Lock-Free Is Actually Faster

Lock-free algorithms beat mutexes when:

1. **High thread count / high contention**: many threads compete for the
   same data.  With a mutex, threads sleep and wake (expensive syscalls).
   With lock-free, threads spin briefly and succeed on the next CAS.

2. **Short critical sections**: the lock-free overhead (CAS retry,
   cache invalidation) is small relative to the work being protected.

3. **Real-time / latency-sensitive code**: a mutex-holding thread can be
   preempted, causing all waiters to stall for a full scheduler quantum
   (e.g. 10 ms).  Lock-free threads are never blocked by a preempted peer.

Lock-free algorithms lose when:

1. **Low thread count**: a single-threaded program wastes time on atomic
   instructions that are more expensive than plain loads/stores.

2. **Long critical sections**: the cost of a failed CAS (cache miss,
   retry loop) exceeds the cost of sleeping on a mutex.

3. **Cache-line ping-pong**: every successful CAS on a shared cache line
   invalidates all other cores' cached copies, forcing an expensive
   cache-coherency protocol message.  With 8+ threads hammering the same
   `q->tail`, throughput can actually fall.

---

## 6. Cache Coherency Cost

When two CPU cores share a cache line (64 bytes on x86/ARM), any write
by one core forces the other cores to **invalidate** their cached copy
(MESI protocol).  The next read by those cores must fetch from the shared
LLC (L3) or main memory.

In the lock-free queue:
- `q->tail` is written by every enqueuer → high coherency traffic.
- `q->head` is written by every dequeuer → same problem.

This is why the benchmark may show the mutex queue winning at low thread
counts: the mutex avoids multiple simultaneous CAS attempts on the same
cache line.
