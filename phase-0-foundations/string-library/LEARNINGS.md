# Learnings — string-library

## 1. Null Termination — Why C Strings Work This Way

A C string is not a data structure with a length field. It's simply a
sequence of bytes in memory that ends with a zero byte (`'\0'`, ASCII value 0).

```
"hello"  →  ['h']['e']['l']['l']['o']['\0']
              0    1    2    3    4    5       ← indices
```

**Why this design?** C was designed in the early 1970s for systems with
kilobytes of RAM.  Storing a length alongside every string would cost 2 extra
bytes per string — a significant overhead when the OS itself fits in 64 KB.
Null termination uses 1 byte per string instead of 2-4.

**The cost:** You must walk the entire string to find its length (O(n)).
This is why `strlen` is O(n), and why it's wasteful to call `strlen` in a loop.

**Contrast with Pascal strings:** Pascal stored the length as the first byte,
making length O(1) but limiting strings to 255 characters.

**Modern practice:** C++ `std::string` and Rust `String` store a length AND
a null terminator — O(1) length lookup, and they can contain embedded nulls.

---

## 2. Why Buffer Overflows Happen

A buffer overflow occurs when you write more bytes than the destination buffer
can hold.

```c
char buf[8];
strcpy(buf, "this string is longer than 8 bytes");  // OVERFLOW
```

The extra bytes overwrite whatever follows `buf` in memory:
- If `buf` is on the stack: this overwrites the saved return address → the
  function "returns" to attacker-controlled code.  This is how 90% of
  classic security vulnerabilities worked.
- If `buf` is on the heap: this overwrites the allocator's metadata or other
  heap objects.

**Why does C allow this?** C does not do bounds checking by default because
checking every array access costs ~5% performance, and C was designed for
systems where that mattered.

**Safe practices:**
- Always use `strncpy`, `snprintf`, `strncat` instead of `strcpy`, `sprintf`, `strcat`
- Track buffer sizes explicitly and pass them to every function that writes
- Use AddressSanitizer (`-fsanitize=address`) during testing
- On modern Linux: stack canaries (`-fstack-protector-all`) and ASLR make
  exploits much harder, but not impossible

---

## 3. The Difference Between `memcpy` and `memmove`

Both copy `n` bytes from `src` to `dst`.  The difference is **overlap safety**.

### `memcpy` — fast, assumes no overlap

`memcpy` copies bytes left-to-right (forward).  If `dst` and `src` overlap and
`dst > src`, the copy corrupts itself:

```
src:  [A B C D E]
dst starts 2 bytes into src:
      [_ _ A B C D E]

Copy A→dst[0], B→dst[1]:  buffer = [A B A B C D E]
Now copy src[2] — but src[2] was overwritten! Reads 'A' instead of 'C'.
```

**Result:** the destination gets wrong data silently.

### `memmove` — correct with any overlap

`memmove` detects the overlap condition:
- If `dst < src`: copy forward (safe — we read before we overwrite)
- If `dst > src` (and they overlap): copy **backward** from the end

```c
if (dst > src && dst < src + n) {
    // Copy backward to avoid clobbering src
    d = dst + n; s = src + n;
    while (n--) *--d = *--s;
}
```

**When to use which:**
- Use `memcpy` when you know the regions don't overlap (it's often faster)
- Use `memmove` when you're not sure, or when implementing things like
  "remove element from the middle of an array" (shifts elements left/right)
- In practice, modern `memcpy` implementations often check for overlap anyway,
  so the performance difference is small

---

## 4. The Dynamic String / Stretchy Buffer Pattern

A fixed-size character array is fine until you don't know how long the
string will be at compile time.

The stretchy buffer solves this:
```c
typedef struct {
    char   *data;  // heap-allocated buffer
    size_t  len;   // current string length
    size_t  cap;   // allocated capacity
} DynStr;
```

**The doubling growth strategy:**

When we need more space, we double the capacity:
```c
while (cap < needed) cap *= 2;
realloc(data, cap);
```

This ensures that `N` append operations cost `O(N)` total, not `O(N^2)`.

*Proof:* Each character can trigger at most one realloc that copies all
existing characters. But after copying N characters, the next realloc doesn't
happen until another N characters are appended.  Total copies ≤ 2N.

Formally: total work = N (appends) + 1 + 2 + 4 + ... + N (copies during
reallocs) < N + 2N = 3N = O(N).

This same analysis applies to `std::vector` in C++, `ArrayList` in Java,
and `list` in Python — all use amortized doubling.
