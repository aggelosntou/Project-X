/*
 * data-structures-c/src/linked_list.h
 *
 * Doubly linked list — each node knows its predecessor and successor.
 *
 * WHY doubly linked (not singly)?
 *   Singly linked lists require O(n) to remove a node given only its pointer,
 *   because you need to find the predecessor to update its ->next.
 *   Doubly linked: O(1) removal given the node pointer — just update
 *   prev->next and next->prev.
 *
 * WHY linked lists over dynamic arrays?
 *   + O(1) insertion/deletion at any position (given a node pointer)
 *   + No reallocation / copying
 *   - O(n) access by index (no random access)
 *   - Poor cache locality (nodes are scattered on the heap)
 *   Use linked lists when you need fast insertion/deletion from the middle
 *   and rarely need random access.
 */

#ifndef LINKED_LIST_H
#define LINKED_LIST_H

#include <stddef.h>

/* Forward declarations */
typedef struct Node Node;
typedef struct List List;

struct Node {
    void  *data;   /* pointer to the item stored in this node */
    Node  *prev;   /* pointer to the previous node (NULL if head) */
    Node  *next;   /* pointer to the next node     (NULL if tail) */
};

struct List {
    Node   *head;  /* first node, or NULL if empty */
    Node   *tail;  /* last node,  or NULL if empty */
    size_t  len;   /* number of nodes */
};

/* Create a new empty list.  Returns NULL on allocation failure. */
List *list_new(void);

/* Insert item at the front (new head).  Returns the new Node, or NULL. */
Node *list_push_front(List *list, void *data);

/* Insert item at the back (new tail).  Returns the new Node, or NULL. */
Node *list_push_back(List *list, void *data);

/* Remove and return the data from the front node.  Returns NULL if empty. */
void *list_pop_front(List *list);

/* Remove and return the data from the back node.  Returns NULL if empty. */
void *list_pop_back(List *list);

/* Remove a specific node from the list (O(1) because it's doubly linked).
 * The caller must free the node's data if needed.
 * Returns the data that was in the node. */
void *list_remove(List *list, Node *node);

/* Find the first node whose data matches 'target' according to cmp.
 * cmp(a, b) should return 0 when a and b are considered equal.
 * Returns the Node, or NULL if not found. */
Node *list_find(const List *list, const void *target,
                int (*cmp)(const void *, const void *));

/* Free the list and all nodes (but NOT the data pointers).
 * The caller is responsible for freeing the stored data. */
void list_free(List *list);

#endif /* LINKED_LIST_H */
