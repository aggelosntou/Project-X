/*
 * pointers-and-memory/src/function_pointers.c
 *
 * PURPOSE: Understand function pointers — pointers to code rather than data.
 *
 * Every function in C has an address (the address of its first machine
 * instruction).  A function pointer stores that address.  You can then
 * call the function indirectly through the pointer.
 *
 * This is the foundation of:
 *   - Callbacks (pass a function to be called later)
 *   - Dispatch tables (choose a function at runtime without if/switch)
 *   - Virtual function tables in C++ (vtables — the mechanism behind polymorphism)
 *   - Plugin architectures (dlopen + dlsym returns a void * you cast to a fn ptr)
 *
 * Compile: gcc -std=c11 -Wall -Wextra -O0 -g -o function_pointers function_pointers.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* -----------------------------------------------------------------------
 * SECTION 1: Basic function pointer syntax
 *
 * The type of a function pointer mirrors the function's signature:
 *
 *   int add(int a, int b) {...}
 *
 *   pointer declaration:   int (*fp)(int, int);
 *                                 ^^           ^^ parameter types
 *                          ^^^ return type
 *                               ^^ name of the pointer variable
 *
 * The extra parentheses around (*fp) are required because without them,
 *   int *fp(int, int)
 * would mean "a function fp that takes two ints and returns int *" —
 * completely different!
 * ----------------------------------------------------------------------- */
static int add(int a, int b) { return a + b; }
static int sub(int a, int b) { return a - b; }
static int mul(int a, int b) { return a * b; }

static void section_basic_syntax(void)
{
    printf("--- Section 1: Basic Function Pointer Syntax ---\n");

    /* Declare fp as a pointer to a function that takes (int, int) and
     * returns int.  Then assign the address of 'add'. */
    int (*fp)(int, int) = add;   /* 'add' here means '&add' — functions
                                   * also decay to pointers, just like arrays */

    printf("  Direct call:   add(3, 4) = %d\n", add(3, 4));
    printf("  Through ptr:   fp(3, 4)  = %d   (same as (*fp)(3, 4))\n", fp(3, 4));

    /* sizeof a function pointer == sizeof any other pointer */
    printf("  sizeof(fp) = %zu bytes (it's just an address)\n", sizeof(fp));

    /* You can reassign it to a different compatible function */
    fp = sub;
    printf("  fp = sub;  fp(10, 3) = %d\n", fp(10, 3));

    fp = mul;
    printf("  fp = mul;  fp(6, 7)  = %d\n", fp(6, 7));

    /* typedef makes the syntax much more readable */
    typedef int (*BinaryOp)(int, int);
    BinaryOp ops[3] = { add, sub, mul };
    const char *names[3] = { "add", "sub", "mul" };
    printf("\n  Using typedef BinaryOp:\n");
    for (int i = 0; i < 3; i++) {
        printf("    %s(10, 3) = %d\n", names[i], ops[i](10, 3));
    }
    printf("\n");
}

/* -----------------------------------------------------------------------
 * SECTION 2: Array of function pointers — dispatch table
 *
 * A dispatch table (jump table) replaces a long if/switch chain.
 *
 * With a switch:
 *   if (op == ADD) return add(a, b);
 *   if (op == SUB) return sub(a, b);
 *   ...
 *
 * With a dispatch table:
 *   return ops[op](a, b);
 *
 * The dispatch table lookup is O(1) and branchless — critical for
 * performance in hot code paths like bytecode interpreters.
 * ----------------------------------------------------------------------- */
typedef enum { OP_ADD = 0, OP_SUB, OP_MUL, OP_COUNT } OpCode;

static int dispatch(int a, int b, OpCode op)
{
    /* The operations array IS the dispatch table.
     * Index directly with the opcode — no if/switch needed. */
    static int (*ops[OP_COUNT])(int, int) = { add, sub, mul };

    if (op >= OP_COUNT) {
        fprintf(stderr, "Unknown opcode: %d\n", (int)op);
        return 0;
    }
    return ops[op](a, b);   /* indirect function call through the table */
}

static void section_dispatch_table(void)
{
    printf("--- Section 2: Dispatch Table ---\n");

    printf("  dispatch(10, 4, OP_ADD) = %d\n", dispatch(10, 4, OP_ADD));
    printf("  dispatch(10, 4, OP_SUB) = %d\n", dispatch(10, 4, OP_SUB));
    printf("  dispatch(10, 4, OP_MUL) = %d\n", dispatch(10, 4, OP_MUL));
    printf("\n  Key benefit: adding a new operation requires only:\n");
    printf("    1. Adding an entry to the enum\n");
    printf("    2. Adding the function to the ops[] array\n");
    printf("  No if/switch to update — the dispatch code is unchanged.\n");
    printf("\n");
}

/* -----------------------------------------------------------------------
 * SECTION 3: Callbacks
 *
 * A callback is a function pointer you pass to another function so it
 * can call your code.  The C standard library uses this pattern:
 *   qsort(array, n, sizeof(element), comparator_function)
 *   bsearch, atexit, signal, etc.
 * ----------------------------------------------------------------------- */

/* A comparator for qsort: returns negative/zero/positive */
static int compare_ints_ascending(const void *a, const void *b)
{
    /* qsort passes void * — we must cast back to the actual type */
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    /* Subtraction is a common trick but overflows for large values.
     * The safe pattern: */
    return (ia > ib) - (ia < ib);
}

static int compare_ints_descending(const void *a, const void *b)
{
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return (ib > ia) - (ib < ia);
}

static void section_callbacks(void)
{
    printf("--- Section 3: Callbacks (qsort example) ---\n");

    int data[] = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    int n = (int)(sizeof(data) / sizeof(data[0]));

    /* Callback pointer has type:  int (*)(const void *, const void *) */
    int (*comparator)(const void *, const void *) = compare_ints_ascending;

    qsort(data, (size_t)n, sizeof(int), comparator);
    printf("  Ascending:  ");
    for (int i = 0; i < n; i++) printf("%d ", data[i]);
    printf("\n");

    comparator = compare_ints_descending;
    qsort(data, (size_t)n, sizeof(int), comparator);
    printf("  Descending: ");
    for (int i = 0; i < n; i++) printf("%d ", data[i]);
    printf("\n\n");
}

/* -----------------------------------------------------------------------
 * SECTION 4: Simulated vtable (C++ virtual dispatch in C)
 *
 * In C++, when you call a virtual method through a base class pointer,
 * the compiler generates code to look up the method in the object's
 * "vtable" — a table of function pointers specific to that class.
 *
 * We can simulate exactly this in C using a struct of function pointers.
 *
 * This is how C libraries implement polymorphism:
 *   - GLib's GObject system
 *   - Linux kernel's file_operations struct
 *   - POSIX's FILE * (each implementation has its own read/write functions)
 * ----------------------------------------------------------------------- */

/* Forward declaration */
typedef struct Shape Shape;

/* The "vtable" — a struct of function pointers.
 * Every "class" that implements Shape will have one of these. */
typedef struct {
    double (*area)(const Shape *);
    double (*perimeter)(const Shape *);
    void   (*describe)(const Shape *);
    void   (*destroy)(Shape *);
} ShapeVTable;

/* The "base class" — every shape starts with a pointer to its vtable */
struct Shape {
    const ShapeVTable *vtable;  /* FIRST member — this is what C++ does too */
    /* (C++ stores a hidden vptr as the first field) */
};

/* --- Circle --- */
typedef struct {
    Shape base;   /* must be first: base class "subobject" */
    double radius;
} Circle;

static double circle_area(const Shape *s) {
    const Circle *c = (const Circle *)s;
    return 3.14159265358979 * c->radius * c->radius;
}
static double circle_perimeter(const Shape *s) {
    const Circle *c = (const Circle *)s;
    return 2.0 * 3.14159265358979 * c->radius;
}
static void circle_describe(const Shape *s) {
    const Circle *c = (const Circle *)s;
    printf("    Circle(r=%.2f): area=%.4f, perimeter=%.4f\n",
           c->radius, circle_area(s), circle_perimeter(s));
}
static void circle_destroy(Shape *s) { free(s); }

static const ShapeVTable circle_vtable = {
    .area      = circle_area,
    .perimeter = circle_perimeter,
    .describe  = circle_describe,
    .destroy   = circle_destroy,
};

static Circle *circle_new(double r) {
    Circle *c = malloc(sizeof(Circle));
    if (!c) return NULL;
    c->base.vtable = &circle_vtable;  /* attach the vtable */
    c->radius = r;
    return c;
}

/* --- Rectangle --- */
typedef struct {
    Shape base;
    double width, height;
} Rectangle;

static double rect_area(const Shape *s) {
    const Rectangle *r = (const Rectangle *)s;
    return r->width * r->height;
}
static double rect_perimeter(const Shape *s) {
    const Rectangle *r = (const Rectangle *)s;
    return 2.0 * (r->width + r->height);
}
static void rect_describe(const Shape *s) {
    const Rectangle *r = (const Rectangle *)s;
    printf("    Rect(%g x %g): area=%.4f, perimeter=%.4f\n",
           r->width, r->height, rect_area(s), rect_perimeter(s));
}
static void rect_destroy(Shape *s) { free(s); }

static const ShapeVTable rect_vtable = {
    .area      = rect_area,
    .perimeter = rect_perimeter,
    .describe  = rect_describe,
    .destroy   = rect_destroy,
};

static Rectangle *rect_new(double w, double h) {
    Rectangle *r = malloc(sizeof(Rectangle));
    if (!r) return NULL;
    r->base.vtable = &rect_vtable;
    r->width  = w;
    r->height = h;
    return r;
}

/* Generic function that works on ANY Shape — this is polymorphism */
static void print_shape_info(Shape *s)
{
    /* This call goes through the vtable:
     *   1. Load s->vtable (the pointer to the function table)
     *   2. Load vtable->describe (the function pointer for describe)
     *   3. Call it
     * This is EXACTLY what C++ does for virtual methods. */
    s->vtable->describe(s);
}

static void section_vtable(void)
{
    printf("--- Section 4: Simulated vtable (C++ virtual dispatch in C) ---\n");

    /* Array of base class pointers — we don't know (or care) what the
     * concrete types are at this point */
    Shape *shapes[3];
    shapes[0] = (Shape *)circle_new(5.0);
    shapes[1] = (Shape *)rect_new(3.0, 4.0);
    shapes[2] = (Shape *)circle_new(1.0);

    printf("  Calling describe() through vtable on each shape:\n");
    for (int i = 0; i < 3; i++) {
        print_shape_info(shapes[i]);
    }

    printf("\n  Each call dispatches to the correct implementation at runtime.\n");
    printf("  This is how C++ virtual functions work under the hood.\n");

    for (int i = 0; i < 3; i++) {
        shapes[i]->vtable->destroy(shapes[i]);
    }
    printf("\n");
}

int main(void)
{
    printf("=== Function Pointers ===\n\n");

    section_basic_syntax();
    section_dispatch_table();
    section_callbacks();
    section_vtable();

    return 0;
}
