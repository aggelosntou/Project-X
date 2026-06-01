/*
 * data-structures-c/src/hash_map.c
 *
 * Implementation of the hash map with separate chaining.
 */

#include "hash_map.h"
#include <stdlib.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * FNV-1a Hash Function
 *
 * FNV-1a (Fowler-Noll-Vo) is a simple non-cryptographic hash function.
 * For each byte: XOR the hash with the byte, then multiply by the prime.
 *
 * The specific prime and offset_basis constants were chosen mathematically
 * to give good avalanche behaviour (small input change → large output change).
 *
 * 64-bit FNV-1a constants:
 *   offset_basis = 14695981039346656037 (0xcbf29ce484222325)
 *   prime        = 1099511628211        (0x100000001b3)
 * ----------------------------------------------------------------------- */
static uint64_t fnv1a_hash(const char *key)
{
    uint64_t hash = 14695981039346656037ULL;  /* offset_basis */
    while (*key != '\0') {
        hash ^= (uint64_t)(unsigned char)*key;
        hash *= 1099511628211ULL;             /* FNV prime */
        key++;
    }
    return hash;
}

/* Needed for uint64_t — included here to keep the header clean */
#include <stdint.h>

/* -----------------------------------------------------------------------
 * Helper: round up to the next power of 2
 * Used so bucket_index = hash & (num_buckets - 1) works correctly.
 * (x & (pow2-1) is equivalent to x % pow2 but branchless and faster)
 * ----------------------------------------------------------------------- */
static size_t next_pow2(size_t n)
{
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

/* -----------------------------------------------------------------------
 * Compute the bucket index for a given hash and num_buckets.
 * (num_buckets is always a power of 2, so we can use &)
 * ----------------------------------------------------------------------- */
static size_t bucket_for(uint64_t hash, size_t num_buckets)
{
    return (size_t)(hash & (num_buckets - 1));
}

HashMap *hm_new(size_t initial_buckets)
{
    HashMap *hm = malloc(sizeof(HashMap));
    if (hm == NULL) return NULL;

    size_t nb = next_pow2(initial_buckets < 8 ? 8 : initial_buckets);

    /* Allocate the bucket array and zero-initialise (all NULL chains) */
    hm->buckets = calloc(nb, sizeof(HMEntry *));
    if (hm->buckets == NULL) {
        free(hm);
        return NULL;
    }

    hm->num_buckets = nb;
    hm->len         = 0;
    return hm;
}

/* -----------------------------------------------------------------------
 * Resize: double the bucket count and rehash all entries.
 *
 * WHY we must rehash (not just memcpy):
 *   bucket_index = hash & (num_buckets - 1)
 *   With 8 buckets: hash=0b1011 → index 3
 *   With 16 buckets: hash=0b1011 → index 11  (different!)
 * So every entry must be moved to its new bucket.
 * ----------------------------------------------------------------------- */
static int hm_resize(HashMap *hm)
{
    size_t new_nb  = hm->num_buckets * 2;
    HMEntry **new_buckets = calloc(new_nb, sizeof(HMEntry *));
    if (new_buckets == NULL) return -1;

    /* Rehash all existing entries into the new bucket array */
    for (size_t i = 0; i < hm->num_buckets; i++) {
        HMEntry *entry = hm->buckets[i];
        while (entry != NULL) {
            HMEntry *next = entry->next;   /* save before we overwrite ->next */

            /* Recompute bucket index for the larger table */
            uint64_t hash = fnv1a_hash(entry->key);
            size_t   new_b = bucket_for(hash, new_nb);

            /* Prepend to the new bucket's chain (order doesn't matter) */
            entry->next      = new_buckets[new_b];
            new_buckets[new_b] = entry;

            entry = next;
        }
    }

    free(hm->buckets);
    hm->buckets     = new_buckets;
    hm->num_buckets = new_nb;
    return 0;
}

int hm_set(HashMap *hm, const char *key, void *value, void **old_value)
{
    /* Resize check FIRST — before computing the bucket, since resize
     * changes num_buckets and thus the bucket index */
    double load = (double)hm->len / (double)hm->num_buckets;
    if (load > 0.75) {
        if (hm_resize(hm) < 0) return -1;
    }

    uint64_t hash = fnv1a_hash(key);
    size_t   b    = bucket_for(hash, hm->num_buckets);

    /* Scan the chain for an existing entry with this key */
    for (HMEntry *e = hm->buckets[b]; e != NULL; e = e->next) {
        if (strcmp(e->key, key) == 0) {
            /* Key exists — update value */
            if (old_value != NULL) *old_value = e->value;
            e->value = value;
            return 0;
        }
    }

    /* Key not found — create a new entry */
    HMEntry *entry = malloc(sizeof(HMEntry));
    if (entry == NULL) return -1;

    /* Store a COPY of the key string.
     * WHY: the caller might free their key string after calling hm_set.
     * We need the key to persist as long as the entry does. */
    entry->key = malloc(strlen(key) + 1);
    if (entry->key == NULL) {
        free(entry);
        return -1;
    }
    strcpy(entry->key, key);
    entry->value = value;

    /* Prepend to the chain (O(1) — prepend is faster than appending) */
    entry->next    = hm->buckets[b];
    hm->buckets[b] = entry;
    hm->len++;

    if (old_value != NULL) *old_value = NULL;
    return 0;
}

void *hm_get(const HashMap *hm, const char *key)
{
    uint64_t hash = fnv1a_hash(key);
    size_t   b    = bucket_for(hash, hm->num_buckets);

    /* Scan the chain for a matching key */
    for (HMEntry *e = hm->buckets[b]; e != NULL; e = e->next) {
        if (strcmp(e->key, key) == 0) {
            return e->value;
        }
    }
    return NULL;  /* not found */
}

int hm_has(const HashMap *hm, const char *key)
{
    uint64_t hash = fnv1a_hash(key);
    size_t   b    = bucket_for(hash, hm->num_buckets);

    for (HMEntry *e = hm->buckets[b]; e != NULL; e = e->next) {
        if (strcmp(e->key, key) == 0) return 1;
    }
    return 0;
}

void *hm_delete(HashMap *hm, const char *key)
{
    uint64_t hash = fnv1a_hash(key);
    size_t   b    = bucket_for(hash, hm->num_buckets);

    HMEntry *prev = NULL;
    HMEntry *curr = hm->buckets[b];

    while (curr != NULL) {
        if (strcmp(curr->key, key) == 0) {
            /* Unlink this entry from the chain */
            if (prev != NULL) {
                prev->next = curr->next;
            } else {
                hm->buckets[b] = curr->next;  /* was the first in chain */
            }

            void *value = curr->value;
            free(curr->key);
            free(curr);
            hm->len--;
            return value;
        }
        prev = curr;
        curr = curr->next;
    }
    return NULL;  /* key not found */
}

size_t hm_len(const HashMap *hm)
{
    return hm->len;
}

void hm_free(HashMap *hm)
{
    if (hm == NULL) return;

    for (size_t i = 0; i < hm->num_buckets; i++) {
        HMEntry *e = hm->buckets[i];
        while (e != NULL) {
            HMEntry *next = e->next;
            free(e->key);
            /* We do NOT free e->value — caller owns the stored data */
            free(e);
            e = next;
        }
    }

    free(hm->buckets);
    free(hm);
}
