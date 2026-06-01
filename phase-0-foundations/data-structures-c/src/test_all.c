/*
 * data-structures-c/src/test_all.c
 *
 * Tests for DynArray, LinkedList, and HashMap using assert().
 *
 * Build:  make
 * Run:    make run
 *
 * If all assertions pass the program prints "All tests passed." and exits 0.
 * If any assertion fails the program prints the failing file/line and aborts.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dynamic_array.h"
#include "linked_list.h"
#include "hash_map.h"

/* =========================================================================
 * Helpers shared by multiple tests
 * ========================================================================= */

/* Compare two int * values — used by da_sort and list_find. */
static int cmp_int_ptr(const void *a, const void *b)
{
    /* a and b are void** (pointers into the da->data array).
     * We dereference once to get the stored void*, then cast to int*. */
    const int *ia = *(const int **)a;
    const int *ib = *(const int **)b;
    return (*ia > *ib) - (*ia < *ib);  /* branchless three-way compare */
}

/* Compare for list_find: a is the node's data (int *), b is the target (int *). */
static int cmp_int_eq(const void *a, const void *b)
{
    return *(const int *)a - *(const int *)b;
}

/* =========================================================================
 * SECTION 1: DynArray tests
 * ========================================================================= */

static void test_dynarray(void)
{
    printf("  [DynArray] push 100 items, get, pop, sort ...\n");

    /* Allocate 100 ints on the heap so we can store void* to them. */
    int *vals[100];
    for (int i = 0; i < 100; i++) {
        vals[i] = malloc(sizeof(int));
        assert(vals[i] != NULL);
        *vals[i] = i;
    }

    /* --- test: push 100 items --- */
    DynArray *da = da_new(4);   /* start small to force multiple resizes */
    assert(da != NULL);
    assert(da_len(da) == 0);

    for (int i = 0; i < 100; i++) {
        int rc = da_push(da, vals[i]);
        assert(rc == 0);
        assert(da_len(da) == (size_t)(i + 1));
    }

    /* --- test: get returns correct values --- */
    for (int i = 0; i < 100; i++) {
        int *v = da_get(da, (size_t)i);
        assert(v == vals[i]);          /* same pointer */
        assert(*v == i);               /* same value   */
    }

    /* --- test: resize happened (capacity must be >= 100) ---
     * We started with cap=4 and pushed 100 items.
     * The capacity doubles: 4→8→16→32→64→128. So cap should be 128. */
    assert(da->cap >= 100);
    assert(da->cap == 128);   /* exact doubling from 4 */

    /* --- test: pop removes last item --- */
    int *last = da_pop(da);
    assert(last == vals[99]);
    assert(*last == 99);
    assert(da_len(da) == 99);

    /* Pop all remaining items */
    for (int i = 98; i >= 0; i--) {
        int *v = da_pop(da);
        assert(v == vals[i]);
        assert(*v == i);
    }
    assert(da_len(da) == 0);

    /* Pop from empty array returns NULL */
    assert(da_pop(da) == NULL);

    /* --- test: sort ---
     * Re-push in reverse order: 99, 98, ..., 0 */
    for (int i = 99; i >= 0; i--) {
        da_push(da, vals[i]);
    }
    assert(da_len(da) == 100);

    /* Verify it's in reverse order before sort */
    assert(*(int *)da_get(da, 0) == 99);
    assert(*(int *)da_get(da, 99) == 0);

    da_sort(da, cmp_int_ptr);

    /* After sort: ascending order */
    for (size_t i = 0; i < 100; i++) {
        int *v = da_get(da, i);
        assert((size_t)*v == i);
    }

    da_free(da);

    /* Free the heap-allocated ints */
    for (int i = 0; i < 100; i++) {
        free(vals[i]);
    }

    printf("  [DynArray] OK\n");
}

/* =========================================================================
 * SECTION 2: LinkedList tests
 * ========================================================================= */

static void test_linkedlist(void)
{
    printf("  [LinkedList] push front/back, remove from middle, find, iterate ...\n");

    /* Allocate ints */
    int a = 10, b = 20, c = 30, d = 40, e = 50;

    List *list = list_new();
    assert(list != NULL);
    assert(list->len == 0);
    assert(list->head == NULL);
    assert(list->tail == NULL);

    /* --- push_back: build 10 → 20 → 30 --- */
    Node *na = list_push_back(list, &a);
    Node *nb = list_push_back(list, &b);
    Node *nc = list_push_back(list, &c);
    assert(na != NULL && nb != NULL && nc != NULL);
    assert(list->len == 3);
    assert(list->head == na);
    assert(list->tail == nc);
    assert(*(int *)na->data == 10);
    assert(*(int *)nc->data == 30);

    /* --- push_front: build 40 → 10 → 20 → 30 --- */
    Node *nd = list_push_front(list, &d);
    assert(nd != NULL);
    assert(list->head == nd);
    assert(list->len == 4);
    assert(*(int *)list->head->data == 40);

    /* --- push_front again: 50 → 40 → 10 → 20 → 30 --- */
    Node *ne = list_push_front(list, &e);
    assert(ne != NULL);
    assert(list->len == 5);
    assert(list->head == ne);
    assert(*(int *)list->head->data == 50);

    /* --- iterate forward and verify order --- */
    int expected_forward[] = {50, 40, 10, 20, 30};
    int idx = 0;
    for (Node *n = list->head; n != NULL; n = n->next) {
        assert(*(int *)n->data == expected_forward[idx]);
        idx++;
    }
    assert(idx == 5);

    /* --- iterate backward and verify order --- */
    int expected_backward[] = {30, 20, 10, 40, 50};
    idx = 0;
    for (Node *n = list->tail; n != NULL; n = n->prev) {
        assert(*(int *)n->data == expected_backward[idx]);
        idx++;
    }
    assert(idx == 5);

    /* --- remove from middle: remove node containing 10 (node na) ---
     * List becomes: 50 → 40 → 20 → 30 */
    void *removed = list_remove(list, na);
    assert(removed == &a);
    assert(*(int *)removed == 10);
    assert(list->len == 4);

    /* Verify neighbours re-linked: nd (40) → nb (20) */
    assert(nd->next == nb);
    assert(nb->prev == nd);

    /* --- remove head (ne, 50): list becomes 40 → 20 → 30 --- */
    removed = list_remove(list, ne);
    assert(removed == &e);
    assert(list->head == nd);
    assert(list->len == 3);
    assert(list->head->prev == NULL);  /* invariant */

    /* --- remove tail (nc, 30): list becomes 40 → 20 --- */
    removed = list_remove(list, nc);
    assert(removed == &c);
    assert(list->tail == nb);
    assert(list->len == 2);
    assert(list->tail->next == NULL);  /* invariant */

    /* --- find --- */
    /* Push some values and find */
    int x = 99, y = 77;
    list_push_back(list, &x);
    list_push_back(list, &y);
    /* List: 40 → 20 → 99 → 77 */

    Node *found = list_find(list, &x, cmp_int_eq);
    assert(found != NULL);
    assert(*(int *)found->data == 99);

    int not_there = 123;
    found = list_find(list, &not_there, cmp_int_eq);
    assert(found == NULL);

    /* --- pop_front / pop_back --- */
    /* Build fresh list: 1 → 2 → 3 */
    List *l2 = list_new();
    int v1 = 1, v2 = 2, v3 = 3;
    list_push_back(l2, &v1);
    list_push_back(l2, &v2);
    list_push_back(l2, &v3);

    void *pf = list_pop_front(l2);
    assert(*(int *)pf == 1);
    assert(l2->len == 2);
    assert(*(int *)l2->head->data == 2);

    void *pb = list_pop_back(l2);
    assert(*(int *)pb == 3);
    assert(l2->len == 1);
    assert(l2->head == l2->tail);  /* single element: head == tail */

    pb = list_pop_back(l2);
    assert(*(int *)pb == 2);
    assert(l2->len == 0);
    assert(l2->head == NULL && l2->tail == NULL);

    /* Pop from empty returns NULL */
    assert(list_pop_front(l2) == NULL);
    assert(list_pop_back(l2)  == NULL);

    list_free(l2);
    list_free(list);

    printf("  [LinkedList] OK\n");
}

/* =========================================================================
 * SECTION 3: HashMap tests
 * ========================================================================= */

/* Simple deterministic string hash for generating test keys */
static void make_key(char *buf, size_t buflen, int n)
{
    snprintf(buf, buflen, "key_%d", n);
}

static void test_hashmap(void)
{
    printf("  [HashMap] set/get/delete, collision handling, resize, 1000 keys ...\n");

    /* --- basic set / get --- */
    HashMap *hm = hm_new(8);
    assert(hm != NULL);
    assert(hm_len(hm) == 0);

    int val_a = 42, val_b = 99;

    int rc = hm_set(hm, "hello", &val_a, NULL);
    assert(rc == 0);
    assert(hm_len(hm) == 1);
    assert(hm_has(hm, "hello") == 1);
    assert(hm_has(hm, "world") == 0);

    int *got = hm_get(hm, "hello");
    assert(got != NULL && *got == 42);

    /* --- update existing key --- */
    void *old_val = NULL;
    rc = hm_set(hm, "hello", &val_b, &old_val);
    assert(rc == 0);
    assert(hm_len(hm) == 1);           /* count unchanged */
    assert(old_val == &val_a);         /* previous value returned */
    got = hm_get(hm, "hello");
    assert(*got == 99);

    /* --- get non-existent key returns NULL --- */
    assert(hm_get(hm, "missing") == NULL);

    /* --- delete --- */
    void *deleted = hm_delete(hm, "hello");
    assert(deleted == &val_b);
    assert(hm_len(hm) == 0);
    assert(hm_has(hm, "hello") == 0);
    assert(hm_get(hm, "hello") == NULL);

    /* delete non-existent returns NULL */
    assert(hm_delete(hm, "hello") == NULL);

    /* --- collision handling ---
     * We force collisions by inserting enough items that some buckets must
     * share chains before the resize threshold is reached.
     * Start with small table (8 buckets), insert 6 items (load = 0.75 exactly,
     * triggers resize at the 7th). */
    int storage[20];
    for (int i = 0; i < 20; i++) storage[i] = i * 100;

    for (int i = 0; i < 6; i++) {
        char key[32];
        make_key(key, sizeof(key), i);
        rc = hm_set(hm, key, &storage[i], NULL);
        assert(rc == 0);
    }
    assert(hm_len(hm) == 6);

    /* Verify all 6 are retrievable (even if collisions occurred) */
    for (int i = 0; i < 6; i++) {
        char key[32];
        make_key(key, sizeof(key), i);
        int *v = hm_get(hm, key);
        assert(v != NULL && *v == i * 100);
    }

    /* --- resize: insert the 7th item, which pushes load > 0.75 on 8 buckets ---
     * After resize the table has 16 buckets. */
    rc = hm_set(hm, "trigger_resize", &storage[10], NULL);
    assert(rc == 0);
    assert(hm_len(hm) == 7);
    assert(hm->num_buckets == 16);   /* exactly doubled */

    /* All previous keys must still be accessible after resize */
    for (int i = 0; i < 6; i++) {
        char key[32];
        make_key(key, sizeof(key), i);
        int *v = hm_get(hm, key);
        assert(v != NULL && *v == i * 100);
    }
    assert(*(int *)hm_get(hm, "trigger_resize") == storage[10]);

    hm_free(hm);

    /* --- 1000 random string keys ---
     * Use a seeded RNG for reproducibility. */
    HashMap *big = hm_new(16);
    assert(big != NULL);

    /* We'll store key → index so we can verify later */
    int indices[1000];
    char keys[1000][32];

    srand(42);
    for (int i = 0; i < 1000; i++) {
        /* Generate a random-ish key that is guaranteed unique per index */
        snprintf(keys[i], sizeof(keys[i]), "rk_%d_%d", i, rand() % 100000);
        indices[i] = i;
        rc = hm_set(big, keys[i], &indices[i], NULL);
        assert(rc == 0);
    }
    assert(hm_len(big) == 1000);

    /* Retrieve all 1000 */
    for (int i = 0; i < 1000; i++) {
        int *v = hm_get(big, keys[i]);
        assert(v != NULL && *v == i);
    }

    /* Delete every other key */
    for (int i = 0; i < 1000; i += 2) {
        void *d = hm_delete(big, keys[i]);
        assert(d == &indices[i]);
    }
    assert(hm_len(big) == 500);

    /* Even-indexed keys gone, odd-indexed keys still present */
    for (int i = 0; i < 1000; i++) {
        if (i % 2 == 0) {
            assert(hm_has(big, keys[i]) == 0);
        } else {
            int *v = hm_get(big, keys[i]);
            assert(v != NULL && *v == i);
        }
    }

    hm_free(big);

    printf("  [HashMap] OK\n");
}

/* =========================================================================
 * main
 * ========================================================================= */

int main(void)
{
    printf("Running data-structures-c tests...\n");

    test_dynarray();
    test_linkedlist();
    test_hashmap();

    printf("\nAll tests passed.\n");
    return 0;
}
