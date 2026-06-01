#ifndef BTREE_H
#define BTREE_H

/*
 * btree.h — B-tree (order 3) implementation.
 *
 * Properties of an order-3 B-tree:
 *   - Every node has at most 2 keys  (MAX_KEYS   = BTREE_ORDER - 1 = 2)
 *   - Every node has at most 3 children (MAX_CHILDREN = BTREE_ORDER = 3)
 *   - Every non-root internal node has at least 1 key (ceil(order/2)-1)
 *   - All leaves are at the same depth
 *
 * This is essentially a 2-3 tree.
 *
 * Operations:
 *   Insert: find leaf, insert key, split if over-full (3 keys → split into
 *           two nodes with 1 key each, push middle key up to parent).
 *   Search: O(log n) key comparisons, descend tree.
 *   Delete: find key, remove; if under-full, try redistribute from sibling,
 *           otherwise merge with sibling (pulls key down from parent).
 *
 * This implementation uses integer keys and long values.
 * In a real database the value would be a record pointer / row ID.
 */

#define BTREE_ORDER   3
#define MAX_KEYS      (BTREE_ORDER - 1)   /* 2 */
#define MAX_CHILDREN  BTREE_ORDER          /* 3 */

typedef struct BTreeNode BTreeNode;

struct BTreeNode {
    int       is_leaf;
    int       num_keys;
    int       keys[MAX_KEYS];
    long      values[MAX_KEYS];        /* stored values (at matching key index) */
    BTreeNode *children[MAX_CHILDREN]; /* children[i] is subtree before keys[i] */
};

typedef struct {
    BTreeNode *root;
} BTree;

/* Allocate a new, empty B-tree. */
BTree *btree_new(void);

/* Insert key/value.  If key already exists, update the value. */
void btree_insert(BTree *t, int key, long value);

/* Search for key.  Returns 1 and sets *out_value if found, 0 otherwise. */
int btree_search(BTree *t, int key, long *out_value);

/* Delete key from the tree.  No-op if key not found. */
void btree_delete(BTree *t, int key);

/* Print the tree in a human-readable format (level-order). */
void btree_print(BTree *t);

/* Free all nodes. */
void btree_free(BTree *t);

/* Return the number of keys stored in the tree. */
int btree_count(BTree *t);

#endif /* BTREE_H */
