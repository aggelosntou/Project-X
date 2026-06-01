/*
 * lfq.c — Michael-Scott Lock-Free Queue (1996)
 *
 * Reference: "Simple, Fast, and Practical Non-Blocking and Blocking
 *             Concurrent Queue Algorithms" — Michael & Scott, PODC 1996.
 *
 * =========================================================================
 * ALGORITHM OVERVIEW
 * =========================================================================
 *
 * The queue is a singly-linked list with a dedicated SENTINEL (dummy) node
 * at the front.  The sentinel is never returned to callers; it sits between
 * head and the first real item.
 *
 *   Initial state:
 *     head ──► [sentinel | next=NULL]
 *     tail ──► [sentinel | next=NULL]
 *
 *   After enqueue("A"):
 *     head ──► [sentinel | next=──►[A | next=NULL]] ◄── tail
 *
 *   After dequeue() → "A":
 *     The old sentinel is freed; the node that held "A" becomes
 *     the new sentinel.
 *     head ──► [A-node (now sentinel) | next=NULL] ◄── tail
 *
 * WHY A SENTINEL?
 *   It separates the head and tail pointers so they can be updated
 *   independently.  Without it, a queue with one element would require
 *   both head and tail to change atomically — impossible with two separate
 *   CAS operations.
 *
 * =========================================================================
 * THE ABA PROBLEM — DETAILED
 * =========================================================================
 *
 * CAS(addr, expected, desired) succeeds when *addr == expected.
 * Consider this scenario without a sentinel / with pointer reuse:
 *
 *   T1: reads head pointer → value A  (node for item "foo")
 *   T2: dequeues "foo" (frees node A), enqueues "bar" — allocator
 *       happens to return the same address for the new node → A again
 *   T1: CAS(&q->head, A, A->next) — SUCCEEDS because *head == A again,
 *       but "A" is now a completely different node.  T1 has swapped in
 *       a dangling or incorrect next pointer.
 *
 * Mitigations in this implementation:
 *   1. The sentinel is never freed while the queue exists; the head
 *      pointer cycles through "real" nodes only, reducing (not
 *      eliminating) reuse risk.
 *   2. For production use, consider: tagged pointers (encode a monotonic
 *      version counter in the unused low bits of the pointer), hazard
 *      pointers (defer frees until no thread holds a reference), or
 *      epoch-based reclamation.
 *
 * =========================================================================
 * MEMORY ORDERING — WHY EACH CHOICE
 * =========================================================================
 *
 * memory_order_acquire on LOADS:
 *   "I need to see all stores that the thread which published this pointer
 *    had done before its release store."  Used when reading head/tail/next
 *    so that we see the fully initialised node, not a torn write.
 *
 * memory_order_release on STORES:
 *   "All my prior stores must be visible to any thread that acquires this
 *    value."  (Not used standalone here; folded into acq_rel on CAS.)
 *
 * memory_order_acq_rel on CAS:
 *   The CAS is simultaneously a store (release) and a load (acquire).
 *   - Release half: the new node is fully written before the CAS publishes
 *     its address — consumers will see a complete node.
 *   - Acquire half: on failure, we re-read the current pointer with
 *     acquire semantics so the retry loop sees fresh data.
 *
 * memory_order_relaxed is NOT used because stale reads of head/tail could
 * cause us to operate on nodes whose writes are not yet globally visible.
 */

#include "lfq.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Internal helpers
 * ---------------------------------------------------------------------- */

static LFQNode *node_alloc(void *data)
{
    LFQNode *n = malloc(sizeof(LFQNode));
    if (!n) {
        /* In a production system you would propagate this error.
         * For this educational implementation we abort. */
        abort();
    }
    atomic_init(&n->next, NULL);
    n->data = data;
    return n;
}

/* -------------------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------------- */

LFQueue *lfq_new(void)
{
    LFQueue *q = malloc(sizeof(LFQueue));
    if (!q) return NULL;

    /* Allocate the sentinel (dummy) node.  Its data field is irrelevant. */
    LFQNode *sentinel = node_alloc(NULL);

    /* Both head and tail start pointing at the sentinel.
     * memory_order_relaxed is fine here: the queue is not yet shared. */
    atomic_init(&q->head, sentinel);
    atomic_init(&q->tail, sentinel);

    return q;
}

/*
 * lfq_enqueue — Michael-Scott enqueue
 *
 * Steps:
 *   1. Allocate a new node.
 *   2. Find the TRUE tail (tail->next == NULL).  If tail is stale
 *      (tail->next != NULL), help the previous enqueuer by swinging
 *      q->tail forward.
 *   3. Atomically link the new node onto the true tail's next pointer.
 *   4. Swing q->tail to point at the new node (best-effort; another
 *      thread may do it first — that is fine).
 */
void lfq_enqueue(LFQueue *q, void *data)
{
    LFQNode *node = node_alloc(data);

    for (;;) {
        /* -- acquire: ensures we see the full state of 'tail' node ------- */
        LFQNode *tail = atomic_load_explicit(&q->tail, memory_order_acquire);

        /* -- acquire: ensures we see the correct next pointer ------------ */
        LFQNode *next = atomic_load_explicit(&tail->next, memory_order_acquire);

        /* Re-read tail to confirm it has not changed since we read it.
         * Without this check we might be working with a snapshot that is
         * already stale, leading to an incorrect CAS. */
        if (tail == atomic_load_explicit(&q->tail, memory_order_acquire)) {

            if (next == NULL) {
                /*
                 * tail->next is NULL → tail is truly the last node.
                 * Try to link our new node.
                 *
                 * acq_rel CAS:
                 *   RELEASE: node is fully written before this pointer
                 *            becomes visible to other threads.
                 *   ACQUIRE (failure path): we re-read tail->next with
                 *            acquire semantics so the loop gets fresh data.
                 */
                if (atomic_compare_exchange_weak_explicit(
                        &tail->next,
                        &next,          /* expected (NULL)                  */
                        node,           /* desired                          */
                        memory_order_acq_rel,
                        memory_order_acquire)) {
                    /* Node is linked.  Now swing tail — best effort.
                     * If this CAS fails, a concurrent enqueuer already
                     * moved tail forward.  Either way, the new node is
                     * reachable. */
                    atomic_compare_exchange_weak_explicit(
                        &q->tail,
                        &tail,
                        node,
                        memory_order_acq_rel,
                        memory_order_acquire);
                    return;
                }
                /* CAS failed: another thread linked its node first.
                 * Retry the outer loop. */
            } else {
                /*
                 * tail->next != NULL: the tail pointer is stale.
                 * Another thread enqueued a node but hasn't swung q->tail
                 * yet.  Help it along — this is the "helping" mechanism
                 * that keeps the algorithm lock-free: even if the original
                 * enqueuer is paused, other enqueuers advance tail.
                 */
                atomic_compare_exchange_weak_explicit(
                    &q->tail,
                    &tail,
                    next,
                    memory_order_acq_rel,
                    memory_order_acquire);
            }
        }
        /* Loop: either our CAS lost the race or tail was stale. */
    }
}

/*
 * lfq_dequeue — Michael-Scott dequeue
 *
 * The sentinel is always at the head.  The "first real item" is in
 * head->next.  We dequeue by advancing head to head->next and reading
 * the data from the new sentinel (which was the old head->next).
 *
 * Steps:
 *   1. Read head, tail, head->next.
 *   2. If queue looks empty (head == tail, next == NULL) → return 0.
 *   3. If tail is lagging (head == tail, next != NULL), help advance it.
 *   4. Otherwise grab the data from next, then CAS head from head to next.
 *   5. On CAS success, free the old sentinel and return 1.
 */
int lfq_dequeue(LFQueue *q, void **out)
{
    for (;;) {
        /* acquire: see full state of head node */
        LFQNode *head = atomic_load_explicit(&q->head, memory_order_acquire);
        /* acquire: see full state of tail node */
        LFQNode *tail = atomic_load_explicit(&q->tail, memory_order_acquire);
        /* acquire: see the node that head->next points to */
        LFQNode *next = atomic_load_explicit(&head->next, memory_order_acquire);

        /* Consistency check: if head changed between our reads, retry. */
        if (head == atomic_load_explicit(&q->head, memory_order_acquire)) {

            if (head == tail) {
                /* Head and tail point at the same node. */
                if (next == NULL) {
                    /* Queue is empty. */
                    return 0;
                }
                /*
                 * Tail is lagging behind a concurrent enqueue.
                 * Help the enqueuer swing tail forward so the next
                 * dequeue attempt will succeed.
                 */
                atomic_compare_exchange_weak_explicit(
                    &q->tail,
                    &tail,
                    next,
                    memory_order_acq_rel,
                    memory_order_acquire);

            } else {
                /*
                 * There is at least one real item: next is it.
                 * Read the data BEFORE the CAS.  After the CAS succeeds,
                 * 'head' (the old sentinel) is freed — we must not touch it.
                 */
                *out = next->data;

                /*
                 * Swing head from the old sentinel to next (which becomes
                 * the new sentinel).
                 *
                 * acq_rel CAS:
                 *   RELEASE: *out is fully written before other threads
                 *            can observe that head advanced.
                 *   ACQUIRE (failure): refresh head for the retry.
                 */
                if (atomic_compare_exchange_weak_explicit(
                        &q->head,
                        &head,          /* expected: old sentinel           */
                        next,           /* desired: new sentinel            */
                        memory_order_acq_rel,
                        memory_order_acquire)) {
                    /*
                     * We won the race.  Free the old sentinel.
                     * NOTE: This is where ABA risk lives in the general case.
                     * If the allocator returns this address before another
                     * thread that still holds a pointer to it finishes, we
                     * have a use-after-free.  Hazard pointers / epoch
                     * reclamation solve this safely.
                     */
                    free(head);
                    return 1;
                }
                /* CAS failed: another consumer got here first.  Retry. */
            }
        }
    }
}

void lfq_free(LFQueue *q)
{
    if (!q) return;

    /* Drain any remaining nodes (including the sentinel). */
    LFQNode *cur = atomic_load_explicit(&q->head, memory_order_relaxed);
    while (cur) {
        LFQNode *nxt = atomic_load_explicit(&cur->next, memory_order_relaxed);
        free(cur);
        cur = nxt;
    }
    free(q);
}
