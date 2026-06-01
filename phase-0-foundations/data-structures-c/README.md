# data-structures-c

Three fundamental data structures implemented from scratch in C, with a full test suite.

## Structures

### Dynamic Array (`dynamic_array.h / .c`)
A generic resizable array, similar to C++ `std::vector`. Stores `void *` pointers so it can hold any type. Capacity doubles whenever the array is full, giving **amortized O(1) push**, O(1) random access, and O(n) resize.

### Doubly Linked List (`linked_list.h / .c`)
A doubly linked list where each node holds `prev` and `next` pointers. Gives **O(1) insertion and deletion** at any position (given a node pointer) and O(n) search. Preferred over arrays when items are frequently inserted or removed from the middle and random access is not needed.

### Hash Map (`hash_map.h / .c`)
A hash map with **separate chaining** for collision resolution. Keys are C strings; values are `void *`. Uses the FNV-1a hash function and resizes (doubles bucket count, rehashes everything) when the load factor exceeds 0.75, keeping average lookup near **O(1)**.

## Build

```
make
```

Compiles all source files and produces the `test_all` binary.

## Run

```
make run
```

Runs the test suite. If all assertions pass you will see:

```
Running data-structures-c tests...
  [DynArray] push 100 items, get, pop, sort ...
  [DynArray] OK
  [LinkedList] push front/back, remove from middle, find, iterate ...
  [LinkedList] OK
  [HashMap] set/get/delete, collision handling, resize, 1000 keys ...
  [HashMap] OK

All tests passed.
```

## Clean

```
make clean
```

Removes the compiled binary.

## File layout

```
src/
  dynamic_array.h   — DynArray API
  dynamic_array.c   — DynArray implementation
  linked_list.h     — List/Node API
  linked_list.c     — List implementation
  hash_map.h        — HashMap API
  hash_map.c        — HashMap implementation
  test_all.c        — Test suite (assert-based)
Makefile
README.md
LEARNINGS.md
```
