/*
 * data-structures-c/src/hash_map.h
 *
 * Hash map with separate chaining for collision resolution.
 *
 * Keys are null-terminated C strings.
 * Values are void * (any pointer type).
 *
 * HOW IT WORKS:
 *   1. Compute hash(key) → an integer
 *   2. Map to a bucket: bucket_index = hash & (num_buckets - 1)
 *      (using power-of-2 bucket count so & is equivalent to % but faster)
 *   3. Each bucket holds a singly linked list of (key, value) pairs
 *   4. To find a key: hash it, go to the bucket, scan the chain for key match
 *
 * WHY separate chaining (not open addressing)?
 *   Chaining is simpler to implement and doesn't degrade badly until
 *   the load factor is much higher.
 *   Open addressing has better cache performance but is trickier to delete from.
 *
 * HASH FUNCTION: FNV-1a (Fowler-Noll-Vo)
 *   Simple, fast, good distribution for strings.
 *   Not cryptographic — use SHA-256 for security-sensitive keys.
 *
 * LOAD FACTOR:
 *   load_factor = len / num_buckets
 *   When load_factor > 0.75, we double the bucket count and rehash everything.
 *   This keeps average chain length at ~0.75, keeping lookups near O(1).
 */

#ifndef HASH_MAP_H
#define HASH_MAP_H

#include <stddef.h>

/* A single key-value entry in a chain */
typedef struct HMEntry HMEntry;
struct HMEntry {
    char    *key;    /* heap-allocated copy of the key string */
    void    *value;  /* the stored value                      */
    HMEntry *next;   /* next entry in the chain (NULL = end)  */
};

typedef struct {
    HMEntry **buckets;     /* array of pointers to chains     */
    size_t    num_buckets; /* must be a power of 2            */
    size_t    len;         /* total number of stored entries  */
} HashMap;

/* Create a new HashMap with the given initial bucket count.
 * initial_buckets will be rounded up to the next power of 2.
 * Returns NULL on allocation failure. */
HashMap *hm_new(size_t initial_buckets);

/* Insert or update key with value.
 * If key already exists, the value is replaced (old value returned via
 * the out-param old_value if non-NULL).
 * Returns 0 on success, -1 on allocation failure. */
int hm_set(HashMap *hm, const char *key, void *value, void **old_value);

/* Retrieve the value for key.
 * Returns the value, or NULL if key is not found.
 * Note: NULL could also be a stored value — use hm_has() if you need
 * to distinguish "key not found" from "key maps to NULL". */
void *hm_get(const HashMap *hm, const char *key);

/* Returns 1 if key exists, 0 otherwise. */
int hm_has(const HashMap *hm, const char *key);

/* Remove key and return its value.  Returns NULL if key not found. */
void *hm_delete(HashMap *hm, const char *key);

/* Current number of stored entries. */
size_t hm_len(const HashMap *hm);

/* Free all memory used by the hash map.
 * Does NOT free the stored values — that's the caller's responsibility. */
void hm_free(HashMap *hm);

#endif /* HASH_MAP_H */
