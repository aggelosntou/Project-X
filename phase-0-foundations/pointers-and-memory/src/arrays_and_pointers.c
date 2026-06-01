/*
 * pointers-and-memory/src/arrays_and_pointers.c
 *
 * PURPOSE: Understand the deep connection between arrays and pointers in C.
 *
 * Key insight: In almost every context, an array NAME "decays" to a
 * pointer to its first element.  This is NOT a conversion — no copy
 * happens.  The array name becomes a pointer to the same memory.
 *
 * The one exception: sizeof(array) gives the TOTAL byte size, not the
 * pointer size.  (This is why sizeof tricks work inside the declaring scope.)
 *
 * Compile: gcc -std=c11 -Wall -Wextra -O0 -g -o arrays_and_pointers arrays_and_pointers.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * SECTION 1: The equivalence a[i] == *(a+i)
 *
 * a[i] is LITERALLY defined as *(a+i) in the C standard.
 * This means a[i] == *(a+i) == *(i+a) == i[a] — and yes, i[a] is valid C!
 * ----------------------------------------------------------------------- */
static void section_equivalence(void)
{
    printf("--- Section 1: a[i] == *(a+i) ---\n");

    int arr[5] = {10, 20, 30, 40, 50};

    for (int i = 0; i < 5; i++) {
        /* These four expressions are IDENTICAL at the machine code level */
        int v1 = arr[i];         /* familiar syntax         */
        int v2 = *(arr + i);     /* explicit pointer math   */
        int v3 = *(i + arr);     /* addition is commutative */
        int v4 = i[arr];         /* bizarre but legal!      */

        printf("  arr[%d] = %d | *(arr+%d) = %d | *(i+arr) = %d | i[arr] = %d\n",
               i, v1, i, v2, v3, v4);
    }
    printf("\n  All four forms produce the same assembly instruction.\n");
    printf("  Prefer arr[i] for readability — the others exist to prove the point.\n");
    printf("\n");
}

/* -----------------------------------------------------------------------
 * SECTION 2: Pointer decay when passing to functions
 *
 * When you pass an array to a function, the compiler automatically
 * converts it to a pointer to the first element.  The function has NO
 * way to know the array's length — you must pass it separately.
 *
 * This is why we always see signatures like:
 *     void process(int *arr, size_t len)
 * rather than:
 *     void process(int arr[])   ← identical to the above!
 * ----------------------------------------------------------------------- */
static void print_array(int *arr, size_t len)
{
    /* Inside this function, arr IS a pointer.  sizeof(arr) == sizeof(int *)
     * NOT sizeof the original array. This is a very common source of bugs. */
    printf("  Inside print_array: sizeof(arr) = %zu  (pointer size, NOT array size!)\n",
           sizeof(arr));

    for (size_t i = 0; i < len; i++) {
        printf("  arr[%zu] = %d\n", i, arr[i]);
    }
}

static void section_decay(void)
{
    printf("--- Section 2: Pointer Decay When Passing to Functions ---\n");

    int data[5] = {1, 2, 3, 4, 5};

    printf("  In main: sizeof(data) = %zu  (total array size: 5 * %zu = %zu bytes)\n",
           sizeof(data), sizeof(int), sizeof(data));
    printf("  Passing to function...\n");

    print_array(data, 5);  /* data decays to &data[0] here */

    /* These are equivalent calls: */
    /* print_array(data, 5)      — array decays to pointer */
    /* print_array(&data[0], 5)  — explicit pointer to first element */

    printf("\n  Rule: always pass the length alongside a decayed array pointer.\n");
    printf("  The idiom: len = sizeof(data)/sizeof(data[0]) works ONLY where\n");
    printf("  data is still a true array (not a decayed pointer).\n");
    printf("\n");
}

/* -----------------------------------------------------------------------
 * SECTION 3: 2D arrays and memory layout
 *
 * A 2D array int a[ROWS][COLS] is stored in ROW-MAJOR order:
 *   a[0][0], a[0][1], ..., a[0][COLS-1], a[1][0], a[1][1], ...
 *
 * All elements are contiguous in memory — there are no pointers between rows.
 * This is called a "true 2D array" and is different from an array of pointers.
 *
 * When a 2D array decays, it becomes a pointer to its FIRST ROW:
 *     int (*p)[COLS] = a;
 * NOT a pointer to int, but a pointer to a WHOLE ROW.
 * ----------------------------------------------------------------------- */

#define ROWS 3
#define COLS 4

static void section_2d_arrays(void)
{
    printf("--- Section 3: 2D Arrays and Memory Layout ---\n");

    /* A true 2D array — all 12 ints are contiguous in memory */
    int matrix[ROWS][COLS];
    int val = 0;
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            matrix[r][c] = ++val;
        }
    }

    printf("  Matrix contents (row-major order in memory):\n");
    for (int r = 0; r < ROWS; r++) {
        printf("  Row %d: ", r);
        for (int c = 0; c < COLS; c++) {
            printf("%3d", matrix[r][c]);
        }
        printf("   (row starts at %p)\n", (void *)matrix[r]);
    }

    /* Verify row-major by casting the whole thing to a 1D pointer */
    int *flat = (int *)matrix;
    printf("\n  Flat view (int *flat = (int*)matrix):\n  ");
    for (int i = 0; i < ROWS * COLS; i++) {
        printf("%3d", flat[i]);
    }
    printf("\n");
    printf("  Row-major: row 0 first, then row 1, then row 2.\n");

    /* Pointer to a row */
    int (*row_ptr)[COLS] = matrix;  /* pointer to an array of COLS ints */
    printf("\n  row_ptr[0][2] = %d  (row 0, col 2)\n", row_ptr[0][2]);
    printf("  row_ptr[1][2] = %d  (row 1, col 2)\n", row_ptr[1][2]);
    printf("  (row_ptr+1) advances by sizeof(int)*COLS = %zu bytes\n",
           sizeof(int) * COLS);

    /* Array of pointers (different from 2D array!) */
    printf("\n  Contrast: array of pointers (NOT a 2D array):\n");
    int *rows[ROWS];
    for (int r = 0; r < ROWS; r++) {
        rows[r] = malloc(COLS * sizeof(int));
        if (rows[r] == NULL) { return; }
        for (int c = 0; c < COLS; c++) {
            rows[r][c] = r * COLS + c + 1;
        }
        printf("    rows[%d] = %p  (heap-allocated, NOT contiguous with other rows)\n",
               r, (void *)rows[r]);
    }
    printf("  Memory is fragmented — rows[0], rows[1], rows[2] may be far apart.\n");
    printf("  True 2D array is faster (better cache locality).\n");

    for (int r = 0; r < ROWS; r++) { free(rows[r]); }
    printf("\n");
}

/* -----------------------------------------------------------------------
 * SECTION 4: const pointer vs pointer to const
 *
 * There are two different things you can make const:
 *   1. The thing pointed TO:   const int *p  (can't change *p through p)
 *   2. The pointer itself:     int * const p  (can't change p itself)
 * ----------------------------------------------------------------------- */
static void section_const_pointers(void)
{
    printf("--- Section 4: const Pointers vs Pointers to const ---\n");

    int x = 10, y = 20;

    /* Pointer to const: you can change WHERE p points, but not *p */
    const int *p_to_const = &x;
    printf("  const int *p_to_const = &x;  *p_to_const = %d\n", *p_to_const);
    p_to_const = &y;   /* OK — changing the pointer */
    printf("  p_to_const = &y;            *p_to_const = %d\n", *p_to_const);
    /* *p_to_const = 99;  ← COMPILE ERROR: can't write through a pointer-to-const */

    /* Const pointer: you can change *p, but not where p points */
    int * const const_p = &x;
    printf("  int * const const_p = &x;   *const_p = %d\n", *const_p);
    *const_p = 42;     /* OK — changing the value */
    printf("  *const_p = 42;              x is now %d\n", x);
    /* const_p = &y;  ← COMPILE ERROR: can't reassign a const pointer */

    printf("\n  For function parameters: 'const int *arr' promises the function\n");
    printf("  will not modify the array through that pointer.\n");
    printf("  This is how library functions (strlen, memcpy src, etc.) signal safety.\n");
    printf("\n");
}

int main(void)
{
    printf("=== Arrays and Pointers ===\n\n");

    section_equivalence();
    section_decay();
    section_2d_arrays();
    section_const_pointers();

    return 0;
}
