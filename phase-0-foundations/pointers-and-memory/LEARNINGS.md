# Learnings — pointers-and-memory

## 1. What a Pointer IS at the Hardware Level

A pointer is just an **unsigned integer** — a memory address.

On a 64-bit system, every pointer is 8 bytes (64 bits), regardless of what
it points to.  `int *`, `char *`, `double *`, and `void *` are all 8 bytes.

The type attached to a pointer is entirely a **compiler fiction** — it exists
only to tell the compiler:
1. How many bytes to read/write when you dereference (`*p`)
2. How many bytes to add when you do pointer arithmetic (`p + 1`)
3. What operations are allowed (type safety)

At the machine code level, a pointer dereference is just a `mov` instruction:
```asm
; int x = *p;  (where p holds address 0x7fff5abc1234)
mov rax, [0x7fff5abc1234]   ; load 4 bytes at that address into rax
```

The CPU knows nothing about types. Types are erased after compilation.

---

## 2. Stack vs Heap — Lifetime and Ownership

The two main memory regions in a C program have fundamentally different
**lifetimes**:

### Stack

- **Automatic**: the compiler manages allocation/deallocation via the stack
  pointer register (RSP on x86-64)
- **Lifetime**: from variable declaration to end of enclosing `{...}` block
- **Direction**: grows **downward** toward lower addresses on x86-64
- **Speed**: O(1) — just `sub rsp, N` to allocate, `add rsp, N` to free
- **Limit**: typically 1–8 MB (set by the OS; ulimit -s shows it)

```c
void f(void) {
    int x = 42;   // x lives on the stack
    // x is valid here
}                  // stack frame popped — x is gone
```

### Heap

- **Dynamic**: you explicitly manage via `malloc` / `free`
- **Lifetime**: from `malloc()` to `free()` — independent of function calls
- **Direction**: grows upward toward higher addresses
- **Speed**: slower than stack (allocator maintains free lists, handles fragmentation)
- **Limit**: bounded only by available virtual memory

```c
int *p = malloc(sizeof(int));  // allocates on heap
// p is valid even after the function returns
free(p);                       // explicit deallocation required
p = NULL;                      // prevent dangling pointer use
```

### The Key Rule

**Never return a pointer to a stack variable.** The caller's use of that
pointer is undefined behaviour — the memory will be reused for the next
function call's frame.

```c
int *bad_function(void) {
    int x = 42;
    return &x;   // UB: x is gone when the function returns
}
```

---

## 3. Why Pointer Arithmetic is in Units of the Pointed-to Type

If `p` is of type `int *` and holds address `0x1000`, then:
- `p + 1` has value `0x1004` (not `0x1001`) because `sizeof(int) = 4`
- `p + 2` has value `0x1008`

This makes arrays work naturally: `p + i` points to the i-th element.

The compiler translates `p + i` into:
```
address = base_address + i * sizeof(*p)
```

This is why you need to know the element type to do pointer arithmetic — the
stride depends on the type size.

`char *` arithmetic always moves by 1 byte (since `sizeof(char) == 1`), which
is why `char *` is used for raw byte manipulation.

---

## 4. What Pointer Decay Is

**Pointer decay** is the implicit conversion of an array expression to a
pointer to its first element.

It happens in almost every context where an array is used:
- Passing an array to a function
- Using an array in arithmetic
- Returning an array from a function (not allowed — it decays first)

```c
int arr[10];
int *p = arr;      // arr decays to &arr[0]
f(arr);            // inside f, arr is an int *, not int[10]
```

**What is NOT decayed:**
- `sizeof(arr)` — gives the total size of the array, not the pointer
- `&arr` — gives a pointer to the whole array (`int (*)[10]`), not `int *`

**Why this matters:**
Once an array has decayed to a pointer, the length information is **lost**.
This is why C array functions always take a separate `size_t len` parameter.
It's also why `sizeof` tricks only work in the scope where the array was declared.

```c
void bad_sizeof(int arr[]) {
    // WRONG: sizeof(arr) == sizeof(int *) == 8, not the array size
    int n = sizeof(arr) / sizeof(arr[0]);  // always 2 on 64-bit!
}
```

---

## 5. Function Pointers and the vtable Pattern

A function pointer holds the **address of the first instruction** of a function.
Calling through a function pointer causes the CPU to jump to that address.

The vtable pattern (used by C++ for virtual dispatch) works like this:
1. Each "class" has a static table of function pointers (the vtable)
2. Each object has a hidden pointer to its class's vtable (the vptr)
3. A virtual call: load vptr → load function pointer from table → call it

This means virtual dispatch costs:
- 2 memory loads (vptr, then the function pointer)
- 1 indirect jump

Compared to a direct function call (1 direct jump), virtual dispatch adds
~2 memory accesses. This is why performance-critical code sometimes avoids
virtual functions (or uses devirtualisation).

The Linux kernel uses this pattern extensively — `struct file_operations`
is a vtable that lets different filesystems (ext4, NFS, tmpfs) implement
the same `read`, `write`, `ioctl` interface.
