/*
 * pointers-and-memory/src/stack_vs_heap.c
 *
 * PURPOSE: Understand the two main memory regions a C program uses,
 * how they grow, and what "lifetime" means for each.
 *
 * THE STACK:
 *   - Managed automatically by the CPU (push/pop via the stack pointer register)
 *   - Grows DOWNWARD in memory (toward lower addresses) on x86-64
 *   - Allocation is O(1): just subtract from the stack pointer (SP)
 *   - Freed automatically when a function returns
 *   - Limited size: typically 1-8 MB (controlled by the OS)
 *   - Variables here are called "automatic storage duration"
 *
 * THE HEAP:
 *   - Managed by the allocator (malloc/free in the C runtime)
 *   - Grows UPWARD (toward higher addresses) in the general case
 *   - Allocation is O(1) amortized but slower than stack (requires bookkeeping)
 *   - Freed ONLY when you call free() — NOT when the function returns
 *   - Size limited only by available virtual memory (gigabytes)
 *   - Variables here are called "dynamic storage duration"
 *
 * Compile: gcc -std=c11 -Wall -Wextra -O0 -g -o stack_vs_heap stack_vs_heap.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* -----------------------------------------------------------------------
 * SECTION 1: Stack frames — visualising how the stack grows
 *
 * Each function call creates a "stack frame" (also called "activation record").
 * The frame contains: local variables, saved registers, return address.
 * When the function returns, the frame is "popped" (SP moves back up).
 *
 * By printing the address of a local variable in nested functions,
 * we can see the stack direction.
 * ----------------------------------------------------------------------- */

/* Three nested functions so we can compare their stack frame addresses */
static void inner(void)
{
    int local_inner = 3;
    printf("  inner()   : &local_inner = %p\n", (void *)&local_inner);
}

static void middle(void)
{
    int local_middle = 2;
    printf("  middle()  : &local_middle = %p\n", (void *)&local_middle);
    inner();
}

static void outer(void)
{
    int local_outer = 1;
    printf("  outer()   : &local_outer  = %p\n", (void *)&local_outer);
    middle();

    printf("\n  Observation: addresses DECREASE as we go deeper.\n");
    printf("  The stack grows DOWNWARD (toward lower addresses) on x86-64.\n");
    printf("  Each function's locals sit at a lower address than its caller's.\n");
}

static void section_stack_frames(void)
{
    printf("--- Section 1: Stack Frames (Call Stack Direction) ---\n");
    int local_main = 0;
    printf("  main()    : &local_main   = %p  (approximate — from demo function)\n",
           (void *)&local_main);
    outer();
    printf("\n");
}

/* -----------------------------------------------------------------------
 * SECTION 2: Stack allocation — what you normally write
 * ----------------------------------------------------------------------- */
static void section_stack_allocation(void)
{
    printf("--- Section 2: Stack Allocation ---\n");

    /* These variables live on the stack.
     * The compiler reserves space for them when it sets up this function's frame.
     * No malloc, no free — the CPU handles it via the stack pointer. */
    int    counter = 0;
    double values[4] = {1.1, 2.2, 3.3, 4.4};
    char   name[16]  = "Alice";

    printf("  counter   @ %p  (stack)\n", (void *)&counter);
    printf("  values[0] @ %p  (stack)\n", (void *)&values[0]);
    printf("  name[0]   @ %p  (stack)\n", (void *)&name[0]);

    /* Elements of values[] are contiguous and descend in address: */
    for (int i = 0; i < 4; i++) {
        printf("    values[%d] @ %p  (offset from values[0]: %+td bytes)\n",
               i, (void *)&values[i],
               (char *)&values[i] - (char *)&values[0]);
    }

    printf("\n  Stack lifetime: these variables cease to exist when\n");
    printf("  section_stack_allocation() returns.  Returning a pointer\n");
    printf("  to a local variable is UB — the memory is reused.\n");

    printf("\n");

    (void)counter;  /* suppress unused-variable warnings */
}

/* -----------------------------------------------------------------------
 * SECTION 3: Heap allocation — malloc / free
 *
 * malloc(n) asks the allocator for n bytes.
 * It returns a void * to the first byte, or NULL on failure.
 * The memory persists until free() is called — regardless of which
 * function calls free().
 * ----------------------------------------------------------------------- */
static void section_heap_allocation(void)
{
    printf("--- Section 3: Heap Allocation ---\n");

    /* Allocate an array of 5 ints on the heap.
     * malloc does NOT initialise the memory — values are garbage.
     * Use calloc() if you need zero-initialisation. */
    int *heap_array = malloc(5 * sizeof(int));
    if (heap_array == NULL) {
        fprintf(stderr, "malloc failed!\n");
        return;
    }

    /* Initialise manually */
    for (int i = 0; i < 5; i++) {
        heap_array[i] = (i + 1) * 10;
    }

    printf("  heap_array (pointer itself) is on the STACK: %p\n",
           (void *)&heap_array);
    printf("  heap_array POINTS TO memory on the HEAP:     %p\n",
           (void *)heap_array);

    for (int i = 0; i < 5; i++) {
        printf("    heap_array[%d] @ %p = %d\n",
               i, (void *)&heap_array[i], heap_array[i]);
    }

    /* realloc: resize an allocation.
     * If there's room after the block, it expands in place.
     * Otherwise it copies to a new location and frees the old one.
     * ALWAYS assign to a NEW variable first — if realloc fails it returns
     * NULL but the original pointer is still valid. */
    int *bigger = realloc(heap_array, 10 * sizeof(int));
    if (bigger == NULL) {
        fprintf(stderr, "realloc failed!\n");
        free(heap_array);
        return;
    }
    heap_array = bigger;  /* safe to overwrite now */

    /* The new elements are uninitialised — fill them */
    for (int i = 5; i < 10; i++) {
        heap_array[i] = (i + 1) * 10;
    }
    printf("\n  After realloc to 10 elements:\n");
    for (int i = 0; i < 10; i++) {
        printf("    heap_array[%d] = %d\n", i, heap_array[i]);
    }

    /* free() returns the memory to the allocator.
     * After free(), the pointer is "dangling" — accessing it is UB.
     * Best practice: set it to NULL immediately after free(). */
    free(heap_array);
    heap_array = NULL;

    printf("\n  Memory freed. heap_array set to NULL to prevent dangling use.\n");
    printf("\n");
}

/* -----------------------------------------------------------------------
 * SECTION 4: What happens when you forget to free (memory leak)
 *
 * The memory is not returned to the OS until the process exits.
 * For long-running programs (servers, daemons) this causes ever-growing
 * memory usage until the process is killed by the OOM killer.
 *
 * Valgrind output for a leak would look like:
 *
 *   HEAP SUMMARY:
 *       in use at exit: 40 bytes in 1 blocks
 *     total heap usage: 1 allocs, 0 frees, 40 bytes allocated
 *
 *   LEAK SUMMARY:
 *      definitely lost: 40 bytes in 1 blocks
 *
 * We do NOT demonstrate an actual leak here — just show what causes one.
 * ----------------------------------------------------------------------- */
static void section_memory_leaks(void)
{
    printf("--- Section 4: Memory Leaks (concept) ---\n");

    printf("  A leak occurs when you malloc() but never free():\n\n");
    printf("    int *p = malloc(40);\n");
    printf("    // ... use p ...\n");
    printf("    // forgot free(p);\n");
    printf("    // function returns — p (the pointer variable) is gone\n");
    printf("    // but the 40 bytes are still allocated with no pointer to them!\n\n");

    printf("  Tools to detect leaks:\n");
    printf("    valgrind --leak-check=full ./stack_vs_heap\n");
    printf("    clang's LeakSanitizer: -fsanitize=address (includes leak detection)\n");
    printf("    Instruments (macOS): the Leaks template\n");

    printf("\n  Rule: every malloc() must have exactly one free().\n");
    printf("  Owner: the entity responsible for calling free() should be\n");
    printf("  clearly documented in your API.\n");
    printf("\n");
}

/* -----------------------------------------------------------------------
 * SECTION 5: Stack vs Heap — comparing addresses
 *
 * On Linux/macOS the address space layout is roughly:
 *   High addresses: stack (grows down ↓)
 *   ...
 *   Low addresses:  heap  (grows up   ↑)
 *   Very low:       code + global data
 *
 * So stack addresses tend to be HIGHER than heap addresses.
 * ----------------------------------------------------------------------- */
static void section_address_comparison(void)
{
    printf("--- Section 5: Stack vs Heap Address Comparison ---\n");

    int   stack_var = 0;
    int  *heap_var  = malloc(sizeof(int));

    if (heap_var == NULL) { return; }

    *heap_var = 0;

    printf("  stack_var @ %p  (on the stack)\n",  (void *)&stack_var);
    printf("  heap_var  @ %p  (on the heap)\n",   (void *)heap_var);

    uintptr_t stack_addr = (uintptr_t)&stack_var;
    uintptr_t heap_addr  = (uintptr_t)heap_var;

    if (stack_addr > heap_addr) {
        printf("  Stack address > Heap address: stack is HIGHER in memory.\n");
        printf("  Difference: ~%zu MB\n",
               (size_t)((stack_addr - heap_addr) / (1024 * 1024)));
    } else {
        printf("  Heap address > Stack address (unusual on this platform).\n");
    }

    free(heap_var);
    heap_var = NULL;
    printf("\n");
}

int main(void)
{
    printf("=== Stack vs Heap ===\n\n");

    section_stack_frames();
    section_stack_allocation();
    section_heap_allocation();
    section_memory_leaks();
    section_address_comparison();

    return 0;
}
