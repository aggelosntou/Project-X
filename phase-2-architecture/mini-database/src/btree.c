/*
 * btree.c — B-tree (order 3) implementation.
 *
 * Notation throughout:
 *   n->keys[0..n->num_keys-1]     are the sorted keys.
 *   n->values[i]                  is the value associated with keys[i].
 *   n->children[0..n->num_keys]   for internal nodes: children[i] is the
 *                                 subtree whose keys are all < keys[i]
 *                                 (and keys[i-1] <= k < keys[i]).
 *
 * Insert strategy: "proactive splitting" — we split full nodes on the way
 * DOWN before we need to, so the leaf insertion never needs to propagate
 * splits back up through a recursion stack.  This keeps the code simpler.
 *
 * Delete strategy:
 *   1. If key is in a leaf: remove it directly, then fix under-full nodes
 *      on the way back up.
 *   2. If key is in an internal node: replace it with its in-order
 *      predecessor (rightmost key of left subtree), then delete that
 *      predecessor from the leaf.
 *   Under-full fix: try to borrow from a sibling; if not possible, merge.
 */

#include "btree.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------
 * Node allocation
 * --------------------------------------------------------------- */
static BTreeNode *node_new(int is_leaf)
{
    BTreeNode *n = (BTreeNode *)calloc(1, sizeof(BTreeNode));
    if (!n) { perror("calloc"); exit(EXIT_FAILURE); }
    n->is_leaf = is_leaf;
    return n;
}

static void node_free(BTreeNode *n)
{
    if (!n) return;
    if (!n->is_leaf) {
        for (int i = 0; i <= n->num_keys; ++i)
            node_free(n->children[i]);
    }
    free(n);
}

/* ---------------------------------------------------------------
 * Search
 * --------------------------------------------------------------- */
static int node_search(BTreeNode *n, int key, long *out_value)
{
    if (!n) return 0;

    /* Find the first key >= key */
    int i = 0;
    while (i < n->num_keys && key > n->keys[i]) ++i;

    if (i < n->num_keys && key == n->keys[i]) {
        if (out_value) *out_value = n->values[i];
        return 1;
    }
    if (n->is_leaf) return 0;
    return node_search(n->children[i], key, out_value);
}

int btree_search(BTree *t, int key, long *out_value)
{
    return node_search(t->root, key, out_value);
}

/* ---------------------------------------------------------------
 * Split: split child n->children[i] (which is full: num_keys == MAX_KEYS+1)
 * into two nodes and push the median key up into n.
 *
 * Before:
 *   n has a slot at position i pointing to a child c with 3 keys.
 *   c->keys = [k0, k1, k2]
 *
 * After:
 *   median key k1 is inserted into n at position i.
 *   c retains k0  (left child, index i).
 *   new node z has k2 (right child, index i+1).
 * --------------------------------------------------------------- */
static void split_child(BTreeNode *parent, int i)
{
    BTreeNode *full_child = parent->children[i];
    int mid = MAX_KEYS;   /* for order 3: mid = 1 (0-indexed) — the single median */

    /* Create new right node */
    BTreeNode *new_right = node_new(full_child->is_leaf);

    /* full_child now has: keys[0..MAX_KEYS] (3 keys total for order 3)
     * We treat num_keys == MAX_KEYS+1 as "overfull" state temporarily
     * inside insert_non_full. The median is at index MAX_KEYS/2 = 1. */
    mid = full_child->num_keys / 2;   /* = 1 for order 3 */

    /* Copy right half of full_child to new_right */
    new_right->num_keys = full_child->num_keys - mid - 1;
    for (int j = 0; j < new_right->num_keys; ++j) {
        new_right->keys[j]   = full_child->keys[mid + 1 + j];
        new_right->values[j] = full_child->values[mid + 1 + j];
    }
    if (!full_child->is_leaf) {
        for (int j = 0; j <= new_right->num_keys; ++j)
            new_right->children[j] = full_child->children[mid + 1 + j];
    }

    /* Grab the median key/value to push up */
    int   median_key   = full_child->keys[mid];
    long  median_value = full_child->values[mid];

    /* Shrink full_child to left half */
    full_child->num_keys = mid;

    /* Make room in parent for the median key */
    for (int j = parent->num_keys; j > i; --j) {
        parent->keys[j]       = parent->keys[j - 1];
        parent->values[j]     = parent->values[j - 1];
        parent->children[j+1] = parent->children[j];
    }

    /* Insert median into parent */
    parent->keys[i]       = median_key;
    parent->values[i]     = median_value;
    parent->children[i+1] = new_right;
    parent->num_keys++;
}

/* ---------------------------------------------------------------
 * Insert into a node that is guaranteed to be non-full.
 * If a child is full when we need to descend into it, split it first.
 * --------------------------------------------------------------- */
static void insert_non_full(BTreeNode *n, int key, long value)
{
    int i = n->num_keys - 1;

    if (n->is_leaf) {
        /* Shift keys right to make room */
        while (i >= 0 && key < n->keys[i]) {
            n->keys[i+1]   = n->keys[i];
            n->values[i+1] = n->values[i];
            --i;
        }
        /* Update if key already exists */
        if (i >= 0 && key == n->keys[i]) {
            n->values[i] = value;
            return;
        }
        n->keys[i+1]   = key;
        n->values[i+1] = value;
        n->num_keys++;
    } else {
        /* Find child to descend into */
        while (i >= 0 && key < n->keys[i]) --i;
        if (i >= 0 && key == n->keys[i]) {
            /* Key exists in this internal node; update value */
            n->values[i] = value;
            return;
        }
        ++i;  /* child index */
        if (n->children[i]->num_keys == MAX_KEYS) {
            split_child(n, i);
            /* After split, the median has been pushed up.
             * Decide which of the two new children to descend into. */
            if (key > n->keys[i])
                ++i;
            else if (key == n->keys[i]) {
                n->values[i] = value;
                return;
            }
        }
        insert_non_full(n->children[i], key, value);
    }
}

/* ---------------------------------------------------------------
 * Public insert
 * --------------------------------------------------------------- */
void btree_insert(BTree *t, int key, long value)
{
    BTreeNode *r = t->root;

    if (r->num_keys == MAX_KEYS) {
        /* Root is full: create new root, make old root a child, split */
        BTreeNode *new_root = node_new(0);
        new_root->children[0] = r;
        new_root->num_keys = 0;
        t->root = new_root;
        split_child(new_root, 0);
        insert_non_full(new_root, key, value);
    } else {
        insert_non_full(r, key, value);
    }
}

/* ---------------------------------------------------------------
 * Delete helpers
 * --------------------------------------------------------------- */

/* Find the largest key in subtree rooted at n */
static int predecessor_key(BTreeNode *n)
{
    while (!n->is_leaf) n = n->children[n->num_keys];
    return n->keys[n->num_keys - 1];
}

static long predecessor_value(BTreeNode *n)
{
    while (!n->is_leaf) n = n->children[n->num_keys];
    return n->values[n->num_keys - 1];
}

/* Find the smallest key in subtree rooted at n */
static int successor_key(BTreeNode *n)
{
    while (!n->is_leaf) n = n->children[0];
    return n->keys[0];
}

static long successor_value(BTreeNode *n)
{
    while (!n->is_leaf) n = n->children[0];
    return n->values[0];
}

/*
 * Ensure child[i] has at least MIN_KEYS (= ceil(order/2)-1 = 1) keys.
 * For order 3, MIN_KEYS = 1.
 *
 * Strategy (in order of preference):
 *   1. Borrow from left sibling if it has > 1 key.
 *   2. Borrow from right sibling if it has > 1 key.
 *   3. Merge child[i] with a sibling (and pull median key down from parent).
 */
#define MIN_KEYS 1

static void fix_child(BTreeNode *parent, int i);

static void delete_from_node(BTreeNode *n, int key);

static void delete_from_node(BTreeNode *n, int key)
{
    if (!n) return;

    int i = 0;
    while (i < n->num_keys && key > n->keys[i]) ++i;

    if (i < n->num_keys && key == n->keys[i]) {
        /* Key found in this node */
        if (n->is_leaf) {
            /* Simple case: just remove from leaf */
            for (int j = i; j < n->num_keys - 1; ++j) {
                n->keys[j]   = n->keys[j+1];
                n->values[j] = n->values[j+1];
            }
            n->num_keys--;
        } else {
            /* Internal node: replace with predecessor or successor */
            if (n->children[i]->num_keys > MIN_KEYS) {
                /* Use predecessor */
                n->keys[i]   = predecessor_key(n->children[i]);
                n->values[i] = predecessor_value(n->children[i]);
                delete_from_node(n->children[i], n->keys[i]);
            } else if (n->children[i+1]->num_keys > MIN_KEYS) {
                /* Use successor */
                n->keys[i]   = successor_key(n->children[i+1]);
                n->values[i] = successor_value(n->children[i+1]);
                delete_from_node(n->children[i+1], n->keys[i]);
            } else {
                /* Both children have min keys: merge children[i] and children[i+1]
                 * pulling keys[i] down into the merged node */
                BTreeNode *left  = n->children[i];
                BTreeNode *right = n->children[i+1];

                /* Pull the separator key down into left */
                left->keys[left->num_keys]   = n->keys[i];
                left->values[left->num_keys] = n->values[i];
                left->num_keys++;

                /* Copy right's keys and children into left */
                for (int j = 0; j < right->num_keys; ++j) {
                    left->keys[left->num_keys]   = right->keys[j];
                    left->values[left->num_keys] = right->values[j];
                    left->num_keys++;
                }
                if (!right->is_leaf) {
                    for (int j = 0; j <= right->num_keys; ++j)
                        left->children[left->num_keys - right->num_keys + j] =
                            right->children[j];
                    /* Recount children pointers correctly */
                    /* left already has correct num_keys; children are indexed 0..num_keys */
                }

                /* Remove separator from parent */
                for (int j = i; j < n->num_keys - 1; ++j) {
                    n->keys[j]       = n->keys[j+1];
                    n->values[j]     = n->values[j+1];
                    n->children[j+1] = n->children[j+2];
                }
                n->num_keys--;
                free(right);

                /* Now delete the key from the merged left child */
                delete_from_node(left, key);
            }
        }
    } else {
        /* Key not in this node */
        if (n->is_leaf) return;  /* key not in tree */

        /* Make sure child[i] has enough keys before descending */
        if (n->children[i]->num_keys == MIN_KEYS)
            fix_child(n, i);

        /* After fix_child, the tree may have changed slightly:
         * re-find the child index for key */
        int ni = 0;
        while (ni < n->num_keys && key > n->keys[ni]) ++ni;
        if (ni < n->num_keys && key == n->keys[ni]) {
            delete_from_node(n, key);  /* key moved up during fix */
        } else {
            delete_from_node(n->children[ni], key);
        }
    }
}

static void fix_child(BTreeNode *parent, int i)
{
    /* Try to borrow from left sibling (children[i-1]) */
    if (i > 0 && parent->children[i-1]->num_keys > MIN_KEYS) {
        BTreeNode *child = parent->children[i];
        BTreeNode *left  = parent->children[i-1];

        /* Shift child keys right */
        for (int j = child->num_keys; j > 0; --j) {
            child->keys[j]   = child->keys[j-1];
            child->values[j] = child->values[j-1];
        }
        if (!child->is_leaf) {
            for (int j = child->num_keys + 1; j > 0; --j)
                child->children[j] = child->children[j-1];
        }

        /* Bring separator key down */
        child->keys[0]   = parent->keys[i-1];
        child->values[0] = parent->values[i-1];
        if (!left->is_leaf)
            child->children[0] = left->children[left->num_keys];
        child->num_keys++;

        /* Promote left sibling's largest key */
        parent->keys[i-1]   = left->keys[left->num_keys - 1];
        parent->values[i-1] = left->values[left->num_keys - 1];
        left->num_keys--;
        return;
    }

    /* Try to borrow from right sibling (children[i+1]) */
    if (i < parent->num_keys && parent->children[i+1]->num_keys > MIN_KEYS) {
        BTreeNode *child  = parent->children[i];
        BTreeNode *right  = parent->children[i+1];

        /* Bring separator key down to end of child */
        child->keys[child->num_keys]   = parent->keys[i];
        child->values[child->num_keys] = parent->values[i];
        if (!right->is_leaf)
            child->children[child->num_keys + 1] = right->children[0];
        child->num_keys++;

        /* Promote right sibling's smallest key */
        parent->keys[i]   = right->keys[0];
        parent->values[i] = right->values[0];

        /* Shift right sibling left */
        for (int j = 0; j < right->num_keys - 1; ++j) {
            right->keys[j]   = right->keys[j+1];
            right->values[j] = right->values[j+1];
        }
        if (!right->is_leaf) {
            for (int j = 0; j < right->num_keys; ++j)
                right->children[j] = right->children[j+1];
        }
        right->num_keys--;
        return;
    }

    /* Merge: use left sibling if available, otherwise right */
    if (i > 0) {
        /* Merge children[i-1] (left) and children[i] (child), pull keys[i-1] down */
        BTreeNode *left  = parent->children[i-1];
        BTreeNode *child = parent->children[i];

        left->keys[left->num_keys]   = parent->keys[i-1];
        left->values[left->num_keys] = parent->values[i-1];
        left->num_keys++;

        for (int j = 0; j < child->num_keys; ++j) {
            left->keys[left->num_keys]   = child->keys[j];
            left->values[left->num_keys] = child->values[j];
            left->num_keys++;
        }
        if (!child->is_leaf) {
            for (int j = 0; j <= child->num_keys; ++j)
                left->children[left->num_keys - child->num_keys + j - 1] =
                    child->children[j];
        }

        for (int j = i - 1; j < parent->num_keys - 1; ++j) {
            parent->keys[j]       = parent->keys[j+1];
            parent->values[j]     = parent->values[j+1];
            parent->children[j+1] = parent->children[j+2];
        }
        parent->num_keys--;
        free(child);
    } else {
        /* Merge children[i] and children[i+1], pull keys[i] down */
        BTreeNode *child = parent->children[i];
        BTreeNode *right = parent->children[i+1];

        child->keys[child->num_keys]   = parent->keys[i];
        child->values[child->num_keys] = parent->values[i];
        child->num_keys++;

        for (int j = 0; j < right->num_keys; ++j) {
            child->keys[child->num_keys]   = right->keys[j];
            child->values[child->num_keys] = right->values[j];
            child->num_keys++;
        }
        if (!right->is_leaf) {
            for (int j = 0; j <= right->num_keys; ++j)
                child->children[child->num_keys - right->num_keys + j - 1] =
                    right->children[j];
        }

        for (int j = i; j < parent->num_keys - 1; ++j) {
            parent->keys[j]       = parent->keys[j+1];
            parent->values[j]     = parent->values[j+1];
            parent->children[j+1] = parent->children[j+2];
        }
        parent->num_keys--;
        free(right);
    }
}

void btree_delete(BTree *t, int key)
{
    if (!t->root || t->root->num_keys == 0) return;

    delete_from_node(t->root, key);

    /* If root has become empty (only possible when it was an internal node
     * with one child that got merged), shrink the tree height */
    if (t->root->num_keys == 0 && !t->root->is_leaf) {
        BTreeNode *old_root = t->root;
        t->root = old_root->children[0];
        free(old_root);
    }
}

/* ---------------------------------------------------------------
 * Count keys (in-order traversal)
 * --------------------------------------------------------------- */
static int count_keys(BTreeNode *n)
{
    if (!n) return 0;
    int c = n->num_keys;
    if (!n->is_leaf)
        for (int i = 0; i <= n->num_keys; ++i)
            c += count_keys(n->children[i]);
    return c;
}

int btree_count(BTree *t)
{
    return count_keys(t->root);
}

/* ---------------------------------------------------------------
 * Print (recursive pre-order with indentation)
 * --------------------------------------------------------------- */
static void print_node(BTreeNode *n, int depth)
{
    if (!n) return;
    for (int d = 0; d < depth; ++d) printf("  ");
    printf("[");
    for (int i = 0; i < n->num_keys; ++i) {
        if (i > 0) printf(", ");
        printf("%d(%ld)", n->keys[i], n->values[i]);
    }
    printf("]\n");
    if (!n->is_leaf)
        for (int i = 0; i <= n->num_keys; ++i)
            print_node(n->children[i], depth + 1);
}

void btree_print(BTree *t)
{
    if (!t->root || t->root->num_keys == 0) {
        printf("(empty tree)\n");
        return;
    }
    print_node(t->root, 0);
}

/* ---------------------------------------------------------------
 * Allocate / free
 * --------------------------------------------------------------- */
BTree *btree_new(void)
{
    BTree *t = (BTree *)malloc(sizeof(BTree));
    if (!t) { perror("malloc"); exit(EXIT_FAILURE); }
    t->root = node_new(1);   /* root starts as an empty leaf */
    return t;
}

void btree_free(BTree *t)
{
    if (!t) return;
    node_free(t->root);
    free(t);
}
