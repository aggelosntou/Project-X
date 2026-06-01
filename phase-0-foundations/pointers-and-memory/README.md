# pointers-and-memory

Four programs that progressively build up understanding of C's memory model,
from raw pointer mechanics to the vtable pattern that underlies C++ polymorphism.

## Files

| File | What it teaches |
|------|-----------------|
| `src/pointers_basics.c` | Declaring, dereferencing, pointer arithmetic, NULL, void *, pointer-to-pointer |
| `src/stack_vs_heap.c` | Stack frame layout (direction), heap malloc/free/realloc, memory leaks |
| `src/arrays_and_pointers.c` | Decay, `a[i]` == `*(a+i)`, 2D arrays, const correctness |
| `src/function_pointers.c` | Syntax, dispatch tables, callbacks, vtable simulation |

## Build and run

```bash
make                    # build all four programs (release, -O2)
make debug              # build all with AddressSanitizer + UBSan

./pointers_basics
./stack_vs_heap
./arrays_and_pointers
./function_pointers
make clean
```

## What to observe

### pointers_basics
- All pointer types have the same size (8 bytes on 64-bit)
- `*(a + i)` and `a[i]` produce identical output
- `i[a]` is also valid — because `a[i]` is defined as `*(a+i)` and addition commutes

### stack_vs_heap
- Watch the addresses printed by `inner()`, `middle()`, `outer()` — they
  *decrease* because the stack grows downward on x86-64
- Stack addresses are much higher than heap addresses (typically >100 MB gap)
- The pointer variable lives on the stack even when it *points to* the heap

### arrays_and_pointers
- `sizeof(data)` inside `main` gives the whole array size; inside `print_array`
  it gives only the pointer size — this is pointer decay in action
- 2D array elements are all contiguous; array-of-pointers rows are scattered

### function_pointers
- Changing a dispatch table entry requires changing one array slot, not every call site
- The vtable section shows *exactly* what C++ generates for virtual methods
