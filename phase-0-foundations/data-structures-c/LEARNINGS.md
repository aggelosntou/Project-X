# LEARNINGS — data-structures-c

## 1. Dynamic Array: amortized O(1) push via capacity doubling

### Why doubling, not +1?

If we grew the array by exactly 1 slot on every push, inserting N items would cost:

```
copies on push 1: 0
copies on push 2: 1   (copy 1 item to new buffer)
copies on push 3: 2
...
copies on push N: N-1
total = 0 + 1 + 2 + ... + (N-1) = N(N-1)/2 = O(N²)
```

That means the _average_ cost per push is O(N) — terrible for a supposedly simple operation.

### The geometric series argument

With doubling, resizes happen at sizes 1, 2, 4, 8, ..., N/2 (assuming N is a power of 2). The copies at each resize equal the old capacity:

```
total copies = 1 + 2 + 4 + ... + N/2 = N - 1   (geometric series, ratio 2)
```

So N pushes cost O(N) total copies → **O(1) amortized per push**.

Any constant growth factor > 1 (e.g. 1.5×, used by some std libraries) gives the same O(1) amortized result; the constant factor changes but the big-O does not.

### Internal vs. external fragmentation

**Internal fragmentation**: memory allocated but not yet used. After pushing 65 items into an array that just doubled from 64 to 128 slots, 63 slots sit empty. In the worst case (just after a resize) up to half the allocated space is unused.

**External fragmentation**: the allocator has enough total free memory but cannot satisfy a request because the free memory is split into many small non-contiguous chunks. A dynamic array combats this by allocating one large contiguous slab rather than many small ones. However, if the heap is fragmented when the array needs to double, `realloc` may have to move the entire array to a new location — O(n) copy — even if there is notionally enough memory.

---

## 2. Linked List: O(1) insert/delete, O(n) search

### Why O(1) insert/delete?

Given a pointer to an existing node, removing it requires only:

```c
node->prev->next = node->next;
node->next->prev = node->prev;
```

Two pointer writes — no shifting, no reallocation. This is why the list is _doubly_ linked: with a singly linked list, removal requires finding the predecessor first (O(n)).

### Why O(n) search?

There is no way to jump to position i. The structure is purely navigational: you must follow pointers from `head` (or `tail`) one step at a time. This means indexed access, binary search, and most sorting algorithms cannot be applied efficiently.

### When to prefer a linked list over a dynamic array

**Prefer a linked list when:**
- You need to insert or delete items frequently in the middle of the collection and you already have a pointer to the location (O(1) vs. O(n) for arrays which must shift elements).
- Items are large structs and copying them on resize would be expensive.
- You need a deque (push/pop at both ends) without the modular-arithmetic complexity of a circular buffer.

**Prefer a dynamic array when:**
- You need random access by index (O(1) vs. O(n)).
- You iterate mostly forward — arrays have excellent cache locality because elements are contiguous; linked list nodes are scattered on the heap, causing many cache misses.
- The data set is small — the per-node malloc overhead and pointer indirection outweigh any theoretical advantage.

---

## 3. Hash Map: FNV-1a, separate chaining, load factor, resize

### FNV-1a hash function

For each byte of the key string:
```
hash = hash XOR byte
hash = hash * FNV_PRIME
```

Starting value (`offset_basis`) and prime are chosen to maximise "avalanche" — a single bit change in the input flips roughly half the bits in the output. This spreads keys evenly across buckets. FNV-1a is _not_ cryptographic: an attacker who can choose keys can craft worst-case inputs. Use SHA-256 or SipHash if keys come from untrusted sources.

### Separate chaining

Each bucket is the head of a singly linked list. When two keys hash to the same bucket they form a chain. Lookup scans the chain with `strcmp`:

```
O(1) average (short chains when load factor is low)
O(n) worst case (all keys in one bucket — adversarial input or terrible hash)
```

### Load factor and why resize at 0.75

```
load_factor = num_entries / num_buckets
```

At load factor 1.0 (one entry per bucket on average), the expected chain length is ~1 by the birthday paradox — still ~O(1), but the constant is large. At 2.0, chains average length 2. Performance degrades roughly linearly with load factor.

**0.75** is the standard trade-off:
- Below 0.75, average chain length ≈ 0.75, so most lookups touch only one or two nodes.
- Above 0.75, lookup time starts growing noticeably.
- 0.75 gives good memory efficiency (not too many empty buckets).

Java's `HashMap` uses the same threshold for the same reason.

### Resize cost

When the load factor exceeds 0.75, we double the bucket count and **rehash every entry**. This is O(n). Because resize happens at most every time the size doubles, the _amortized_ cost per insert is O(1) (same geometric series argument as the dynamic array).

After resize every entry must be re-placed: the bucket index is `hash & (num_buckets - 1)`. With 8 buckets, `hash=0b1011` maps to index 3. With 16 buckets, the same hash maps to index 11. Bucket assignments change, so we cannot just copy the old bucket array.

### Worst case O(n) and when it occurs

If all N keys land in the same bucket (all have the same `hash & (num_buckets - 1)`), every operation scans a chain of length N → O(n). This happens when:
1. The hash function is bad (many collisions for common key patterns), or
2. An attacker deliberately crafts key sets that collide (hash-flooding attack).

Mitigations: randomise the hash seed at startup (adds unpredictability), use a collision-resistant hash like SipHash, or switch to open addressing with robin-hood probing.
