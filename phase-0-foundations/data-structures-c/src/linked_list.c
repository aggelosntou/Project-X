/*
 * data-structures-c/src/linked_list.c
 *
 * Implementation of the doubly linked list.
 *
 * Invariant maintained throughout all operations:
 *   - list->head->prev == NULL
 *   - list->tail->next == NULL
 *   - For every non-tail node n: n->next->prev == n
 *   - For every non-head node n: n->prev->next == n
 *   - list->len == number of nodes
 */

#include "linked_list.h"
#include <stdlib.h>

List *list_new(void)
{
    List *list = malloc(sizeof(List));
    if (list == NULL) return NULL;
    list->head = NULL;
    list->tail = NULL;
    list->len  = 0;
    return list;
}

/* Internal helper: create a new node with the given data.
 * Does NOT link it to any list. */
static Node *node_new(void *data)
{
    Node *n = malloc(sizeof(Node));
    if (n == NULL) return NULL;
    n->data = data;
    n->prev = NULL;
    n->next = NULL;
    return n;
}

Node *list_push_front(List *list, void *data)
{
    Node *n = node_new(data);
    if (n == NULL) return NULL;

    if (list->head == NULL) {
        /* List is empty — n is both head and tail */
        list->head = n;
        list->tail = n;
    } else {
        /* Link n before the current head:
         *   n <-> old_head <-> ... <-> tail */
        n->next          = list->head;
        list->head->prev = n;
        list->head       = n;
    }

    list->len++;
    return n;
}

Node *list_push_back(List *list, void *data)
{
    Node *n = node_new(data);
    if (n == NULL) return NULL;

    if (list->tail == NULL) {
        /* List is empty */
        list->head = n;
        list->tail = n;
    } else {
        /* Link n after the current tail:
         *   head <-> ... <-> old_tail <-> n */
        n->prev          = list->tail;
        list->tail->next = n;
        list->tail       = n;
    }

    list->len++;
    return n;
}

void *list_pop_front(List *list)
{
    if (list->head == NULL) return NULL;

    Node *n    = list->head;
    void *data = n->data;

    list->head = n->next;
    if (list->head != NULL) {
        list->head->prev = NULL;  /* new head has no predecessor */
    } else {
        list->tail = NULL;        /* list became empty */
    }

    free(n);
    list->len--;
    return data;
}

void *list_pop_back(List *list)
{
    if (list->tail == NULL) return NULL;

    Node *n    = list->tail;
    void *data = n->data;

    list->tail = n->prev;
    if (list->tail != NULL) {
        list->tail->next = NULL;  /* new tail has no successor */
    } else {
        list->head = NULL;        /* list became empty */
    }

    free(n);
    list->len--;
    return data;
}

void *list_remove(List *list, Node *node)
{
    /* This is the key advantage of doubly linked lists:
     * removing a node we already have a pointer to is O(1).
     * We just update the neighbours' pointers. */

    void *data = node->data;

    if (node->prev != NULL) {
        node->prev->next = node->next;   /* bypass node */
    } else {
        list->head = node->next;         /* node was the head */
    }

    if (node->next != NULL) {
        node->next->prev = node->prev;   /* bypass node */
    } else {
        list->tail = node->prev;         /* node was the tail */
    }

    free(node);
    list->len--;
    return data;
}

Node *list_find(const List *list, const void *target,
                int (*cmp)(const void *, const void *))
{
    /* Linear scan — there's no way around O(n) for unsorted linked lists */
    for (Node *n = list->head; n != NULL; n = n->next) {
        if (cmp(n->data, target) == 0) {
            return n;
        }
    }
    return NULL;
}

void list_free(List *list)
{
    if (list == NULL) return;

    Node *n = list->head;
    while (n != NULL) {
        Node *next = n->next;
        /* We free the Node struct but NOT n->data.
         * The caller owns the data and is responsible for its lifetime. */
        free(n);
        n = next;
    }

    free(list);
}
