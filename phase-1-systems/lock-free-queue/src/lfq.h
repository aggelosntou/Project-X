#ifndef LFQ_H
#define LFQ_H

/*
 * lfq.h — Lock-Free Queue (Michael-Scott, 1996)
 *
 * A concurrent FIFO queue that guarantees progress without locks.
 * Uses C11 _Atomic and atomic_compare_exchange operations.
 *
 * MEMORY ORDERING PRIMER:
 *   memory_order_acquire  — "I will see all stores that happened before
 *                            the release that published this value."
 *   memory_order_release  — "All my prior stores are visible to any
 *                            thread that acquires this value."
 *   memory_order_acq_rel  — Both: this is a store AND a load (used by CAS).
 *
 * ABA PROBLEM:
 *   Thread T1 reads pointer value A from a shared location.
 *   Thread T2 removes A, does work, re-inserts a *new* node that happens
 *   to land at the same address A (e.g. allocator reuse).
 *   T1's CAS succeeds — it "sees" A and swaps — but the invariant it was
 *   relying on (A is the same node) is broken.
 *
 *   In this queue the dummy/sentinel approach partially mitigates ABA:
 *   the sentinel is never returned to the allocator while the queue is
 *   live, so the head pointer cannot cycle back to a previously-freed
 *   sentinel.  However, general node re-use can still cause ABA on the
 *   tail pointer.  Production solutions use tagged pointers (storing a
 *   version counter in the low bits) or hazard pointers to defer frees.
 */

#include <stdatomic.h>
#include <stddef.h>

typedef struct LFQNode LFQNode;
struct LFQNode {
    _Atomic(LFQNode *) next; /* next node; NULL when this is the last node */
    void              *data; /* opaque payload (NULL for sentinel)          */
};

typedef struct {
    _Atomic(LFQNode *) head; /* consumer side — points at the sentinel      */
    _Atomic(LFQNode *) tail; /* producer side — points at last real node     */
} LFQueue;

/*
 * lfq_new   — Allocate and initialise a queue with a sentinel node.
 *             Returns NULL on allocation failure.
 */
LFQueue *lfq_new(void);

/*
 * lfq_enqueue — Push data onto the tail of the queue.
 *               Never blocks; busy-spins only if another thread is
 *               mid-enqueue (extremely brief).
 */
void lfq_enqueue(LFQueue *q, void *data);

/*
 * lfq_dequeue — Pop the head item into *out.
 *               Returns 1 on success, 0 if the queue was empty.
 *               Never blocks.
 */
int lfq_dequeue(LFQueue *q, void **out);

/*
 * lfq_free — Drain the queue and free all memory including the sentinel.
 *            Not thread-safe; call only after all producers/consumers stop.
 */
void lfq_free(LFQueue *q);

#endif /* LFQ_H */
