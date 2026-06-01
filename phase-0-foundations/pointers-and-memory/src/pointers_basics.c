/*
 * pointers-and-memory/src/pointers_basics.c
 *
 * PURPOSE: Understand pointers from the ground up.
 *
 * A pointer is just an INTEGER that holds a MEMORY ADDRESS.
 * On a 64-bit system, every pointer is 8 bytes — regardless of what it
 * points to.  The TYPE of a pointer tells the compiler how many bytes to
 * read/write when you dereference it, and how many bytes to advance when
 * you do pointer arithmetic.
 *
 * Compile: gcc -std=c11 -Wall -Wextra -O0 -g -o pointers_basics pointers_basics.c
 */

#include <stdio.h>
#include <stdlib.h>  /* for NULL */
#include <stdint.h>  /* for uintptr_t */

/* -----------------------------------------------------------------------
 * SECTION 1: Declaring and dereferencing pointers
 * ----------------------------------------------------------------------- */
static void section_declaring(void)
{
    printf("--- Section 1: Declaring and Dereferencing ---\n");

    int x = 42;

    /* int *p  declares a variable named p whose TYPE is "pointer to int".
     * The * is part of the TYPE, not the name.  p itself is stored on the
     * stack just like x — it just holds an address rather than a value. */
    int *p = &x;   /* & is the "address-of" operator — gives the address of x */

    printf("  x      = %d\n", x);
    printf("  &x     = %p  (the address of x in memory)\n", (void *)&x);
    printf("  p      = %p  (p stores that same address)\n", (void *)p);
    printf("  *p     = %d  (* is the dereference operator: follow the address)\n", *p);
    printf("  sizeof(p) = %zu bytes (ALL pointers are the same size on 64-bit)\n",
           sizeof(p));

    /* Modifying x through the pointer */
    *p = 100;
    printf("  After *p = 100: x = %d\n", x);

    printf("\n");
}

/* -----------------------------------------------------------------------
 * SECTION 2: Pointer arithmetic
 *
 * When you do p + 1, you do NOT add 1 to the raw address.
 * You add sizeof(*p) bytes — so the result points to the NEXT element
 * of whatever type p points to.
 *
 * This is why arrays work: elements are contiguous in memory, and
 * incrementing the pointer moves to the next one.
 * ----------------------------------------------------------------------- */
static void section_arithmetic(void)
{
    printf("--- Section 2: Pointer Arithmetic ---\n");

    int arr[5] = {10, 20, 30, 40, 50};
    int *p = arr;  /* arr decays to a pointer to its first element */

    printf("  sizeof(int) = %zu bytes\n", sizeof(int));
    printf("  arr starts at address: %p\n", (void *)arr);

    for (int i = 0; i < 5; i++) {
        /* p + i does NOT add i bytes; it adds i * sizeof(int) bytes.
         * This is pointer arithmetic in units of the pointed-to type. */
        printf("  p + %d = %p, *(p+%d) = %d  (offset in bytes: %td)\n",
               i, (void *)(p + i), i, *(p + i),
               (char *)(p + i) - (char *)p);
    }

    /* Cast to uintptr_t (an integer type guaranteed to hold a pointer)
     * to do raw arithmetic and see the actual byte offsets */
    uintptr_t base = (uintptr_t)p;
    printf("\n  Verifying: each element is %zu bytes after the previous:\n",
           sizeof(int));
    for (int i = 0; i < 5; i++) {
        printf("    arr[%d] at byte offset %td from base\n",
               i, (ptrdiff_t)((uintptr_t)(p + i) - base));
    }

    printf("\n");
}

/* -----------------------------------------------------------------------
 * SECTION 3: The NULL pointer
 *
 * NULL is defined as (void *)0 — the zero address.
 * The OS ensures address 0 is never mapped, so dereferencing NULL always
 * crashes (segmentation fault).  This is INTENTIONAL: a crash is better
 * than silently corrupting random memory.
 *
 * Always check pointers from external sources (malloc, function args)
 * before dereferencing.
 * ----------------------------------------------------------------------- */
static void section_null(void)
{
    printf("--- Section 3: NULL Pointer ---\n");

    int *p = NULL;

    printf("  p = %p  (NULL — the zero address)\n", (void *)p);
    printf("  (int *)NULL == 0: %s\n", (p == NULL) ? "true" : "false");

    /* Check before use — this is the correct pattern */
    if (p != NULL) {
        printf("  *p = %d\n", *p);
    } else {
        printf("  p is NULL — skipping dereference to avoid crash\n");
    }

    /* WHY does dereferencing NULL crash?
     * The OS marks page 0 as non-readable/non-writable.
     * Any access triggers a hardware exception → OS sends SIGSEGV → crash.
     * The crash is a FEATURE: it tells you exactly where the bug is. */
    printf("  Dereferencing NULL would trigger SIGSEGV (segfault).\n");

    printf("\n");
}

/* -----------------------------------------------------------------------
 * SECTION 4: void pointer
 *
 * void * is a "pointer to unspecified type" — a raw address with no
 * type information attached.  You cannot dereference it directly.
 * You must cast it to the correct type first.
 *
 * malloc() returns void * because it doesn't know what you're allocating.
 * memcpy() takes void * so it can work with any type.
 * ----------------------------------------------------------------------- */
static void section_void_pointer(void)
{
    printf("--- Section 4: void Pointer ---\n");

    int   x = 42;
    float f = 3.14f;
    char  c = 'A';

    /* void * can hold ANY pointer type without a cast */
    void *vp;

    vp = &x;
    printf("  vp points to int, value = %d\n", *(int *)vp);   /* must cast */

    vp = &f;
    printf("  vp points to float, value = %.2f\n", *(float *)vp);

    vp = &c;
    printf("  vp points to char, value = '%c'\n", *(char *)vp);

    /* sizeof(void *) is the same as sizeof(any other pointer) */
    printf("  sizeof(void *) = %zu, sizeof(int *) = %zu, sizeof(char *) = %zu\n",
           sizeof(void *), sizeof(int *), sizeof(char *));

    printf("\n");
}

/* -----------------------------------------------------------------------
 * SECTION 5: Pointer to pointer
 *
 * A pointer to pointer (double pointer, **) is exactly what it sounds like:
 * a variable that holds the address of another pointer variable.
 *
 * This is used for:
 *   1. argv in main (array of pointers to strings: char *argv[] = char **)
 *   2. Modifying a pointer that was passed to a function
 *   3. 2D arrays allocated on the heap
 * ----------------------------------------------------------------------- */
static void section_pointer_to_pointer(void)
{
    printf("--- Section 5: Pointer to Pointer ---\n");

    int  x  = 99;
    int *p  = &x;    /* p  holds the address of x */
    int **pp = &p;   /* pp holds the address of p */

    printf("  x   = %d\n", x);
    printf("  p   = %p  (address of x)\n", (void *)p);
    printf("  pp  = %p  (address of p itself)\n", (void *)pp);
    printf("  *p  = %d  (value at address stored in p = x)\n", *p);
    printf("  *pp = %p  (value at address stored in pp = p)\n", (void *)*pp);
    printf("  **pp= %d  (dereference twice: follow pp to p, then p to x)\n", **pp);

    /* Modifying x through pp */
    **pp = 200;
    printf("  After **pp = 200: x = %d, *p = %d\n", x, *p);

    /* Common use case: a function that modifies a pointer */
    printf("\n  [Common use case: modifying a pointer inside a function]\n");
    printf("  void allocate(int **out) { *out = malloc(sizeof(int)); }\n");
    printf("  Without **, the function gets a COPY of the pointer and\n");
    printf("  any assignment is invisible to the caller.\n");

    printf("\n");
}

int main(void)
{
    printf("=== Pointers Basics ===\n\n");

    section_declaring();
    section_arithmetic();
    section_null();
    section_void_pointer();
    section_pointer_to_pointer();

    return 0;
}
