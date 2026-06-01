/*
 * hello-systems/src/main.c
 *
 * PURPOSE: Introduce the fundamental mechanics of a C program as seen
 * by the operating system — arguments, environment, exit codes, and
 * the macros the preprocessor inserts at compile time.
 *
 * Build variants (see Makefile):
 *   make         → release (-O2 -Wall -Wextra)
 *   make debug   → debug  (-g -O0 -fsanitize=address)
 *   make strict  → strict (-Wall -Wextra -Werror -pedantic)
 *   make asm     → emit assembly listing (main.s)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * COMPILE-TIME MACROS
 *
 * The C preprocessor runs BEFORE the compiler sees any source code.
 * These macros are filled in by the preprocessor — they are not
 * runtime values. They exist purely as text substitution.
 *
 * __FILE__   → the source file name (a string literal)
 * __LINE__   → the current line number (an integer literal)
 * __DATE__   → the date the translation unit was compiled ("Jun 01 2026")
 * __TIME__   → the time of compilation ("14:32:07")
 * __func__   → the current function name (C99 and later)
 *
 * WHY this matters: when you ship a binary you can embed the exact
 * commit / build timestamp inside it so a crash report can tell you
 * exactly which binary produced it.
 * ----------------------------------------------------------------------- */
static void print_build_info(void)
{
    printf("=== Build Information ===\n");
    printf("  Source file : %s\n", __FILE__);
    printf("  Compiled on : %s at %s\n", __DATE__, __TIME__);
    printf("  C standard  : %ld\n", (long)__STDC_VERSION__);

    /* Compiler-specific macros — only defined when using GCC or Clang */
#if defined(__GNUC__) && !defined(__clang__)
    printf("  Compiler    : GCC %d.%d.%d\n",
           __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#elif defined(__clang__)
    printf("  Compiler    : Clang %d.%d.%d\n",
           __clang_major__, __clang_minor__, __clang_patchlevel__);
#else
    printf("  Compiler    : unknown\n");
#endif

    /* Detect optimisation level via GCC's __OPTIMIZE__ */
#ifdef __OPTIMIZE__
    printf("  Optimised   : yes\n");
#else
    printf("  Optimised   : no (debug build)\n");
#endif
    printf("\n");
}

/* -----------------------------------------------------------------------
 * ARGUMENT PARSING
 *
 * Every C program's entry point is:
 *     int main(int argc, char *argv[])
 *
 * argc  = argument COUNT  (always >= 1; argv[0] is the program name)
 * argv  = argument VECTOR (NULL-terminated array of C strings)
 *
 * The shell splits the command line on whitespace and passes each token
 * as a separate element.  So:
 *     ./hello foo bar   →  argc=3, argv={"./hello","foo","bar",NULL}
 * ----------------------------------------------------------------------- */
static void demonstrate_arguments(int argc, char *argv[])
{
    printf("=== Command-Line Arguments ===\n");
    printf("  argc = %d\n", argc);

    for (int i = 0; i < argc; i++) {
        /* argv[i] is a pointer to a null-terminated character array */
        printf("  argv[%d] = \"%s\"\n", i, argv[i]);
    }

    /* A common pattern: scan for a known flag */
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            verbose = 1;
        }
    }
    printf("  --verbose flag: %s\n", verbose ? "set" : "not set");
    printf("\n");
}

/* -----------------------------------------------------------------------
 * ENVIRONMENT VARIABLES
 *
 * The OS passes a "environment" to every process: an array of
 * "KEY=VALUE" strings.  getenv() does a linear search through this
 * array and returns a pointer to the value portion, or NULL.
 *
 * WHY this matters: environment variables are the standard way to
 * configure programs without recompilation (database URLs, API keys,
 * log levels, etc.).
 * ----------------------------------------------------------------------- */
static void demonstrate_env(void)
{
    printf("=== Environment Variables ===\n");

    /* getenv returns a pointer INSIDE the environment block.
     * Do NOT free it; do NOT write to it.  If you need to modify,
     * copy it first with strdup(). */
    const char *home = getenv("HOME");
    const char *path = getenv("PATH");
    const char *user = getenv("USER");
    const char *missing = getenv("VARIABLE_THAT_DOES_NOT_EXIST");

    printf("  HOME  = %s\n", home  ? home  : "(not set)");
    printf("  USER  = %s\n", user  ? user  : "(not set)");

    /* PATH can be very long; truncate for readability */
    if (path) {
        /* snprintf is safer than sprintf — it never writes past the buffer */
        char short_path[64];
        snprintf(short_path, sizeof(short_path), "%.60s...", path);
        printf("  PATH  = %s\n", short_path);
    }

    printf("  MISSING = %s\n", missing ? missing : "(not set — getenv returned NULL)");
    printf("\n");
}

/* -----------------------------------------------------------------------
 * EXIT CODES
 *
 * When main() returns (or exit() is called), that integer is the
 * "exit status" returned to the parent process (your shell).
 *
 *   0          → success  (EXIT_SUCCESS is defined as 0 in <stdlib.h>)
 *   1..127     → failure  (1 is the generic "error" by convention)
 *   128+N      → killed by signal N (set by the shell, not your code)
 *
 * In the shell:  $?  holds the exit status of the last command.
 *
 * WHY this matters: scripts check exit codes to decide whether to
 * continue.  A wrong exit code silently breaks pipelines.
 * ----------------------------------------------------------------------- */
static void demonstrate_exit_codes(void)
{
    printf("=== Exit Codes ===\n");
    printf("  EXIT_SUCCESS = %d\n", EXIT_SUCCESS);
    printf("  EXIT_FAILURE = %d\n", EXIT_FAILURE);
    printf("  This program will exit with EXIT_SUCCESS (0).\n");
    printf("  Try:  ./hello ; echo \"Exit status: $?\"\n");
    printf("  Or :  ./hello --fail ; echo \"Exit status: $?\"\n");
    printf("\n");
}

/* -----------------------------------------------------------------------
 * UNDEFINED BEHAVIOUR (UB)
 *
 * The C standard leaves certain operations "undefined" — meaning the
 * compiler is free to do ANYTHING: crash, produce wrong results,
 * nasal demons, or appear to work fine.
 *
 * WHY C has UB: because it lets the compiler optimise more aggressively.
 * If signed integer overflow is defined (e.g., wraps), the compiler
 * cannot assume (x + 1 > x) is always true.  UB lets it make that
 * assumption, enabling optimisations.
 *
 * The commented code below WOULD be UB if uncommented:
 * ----------------------------------------------------------------------- */
static void demonstrate_ub_concept(void)
{
    printf("=== Undefined Behaviour (concept) ===\n");
    printf("  The following examples are commented out — they are UB.\n\n");

    printf("  // UB example 1: signed integer overflow\n");
    printf("  // int x = INT_MAX;\n");
    printf("  // int y = x + 1;  // UB: signed overflow wraps on most CPUs\n");
    printf("  //                 // but the C standard says the result is undefined\n\n");

    printf("  // UB example 2: out-of-bounds array access\n");
    printf("  // int a[5] = {0};\n");
    printf("  // int v = a[10]; // UB: reads memory past the array\n\n");

    printf("  // UB example 3: dereferencing a NULL pointer\n");
    printf("  // int *p = NULL;\n");
    printf("  // *p = 42;       // UB: almost always a segfault, but not guaranteed\n\n");

    printf("  // UB example 4: using memory after free\n");
    printf("  // int *q = malloc(sizeof(int));\n");
    printf("  // free(q);\n");
    printf("  // *q = 1;       // UB: the allocator may have reused this memory\n\n");

    printf("  Compile with -fsanitize=address,undefined to catch UB at runtime.\n");
    printf("\n");
}

/* -----------------------------------------------------------------------
 * MAIN — the entry point
 *
 * The OS calls main() after setting up the stack, heap, and loading
 * shared libraries.  The return value goes to the shell as the exit code.
 * ----------------------------------------------------------------------- */
int main(int argc, char *argv[])
{
    /* Check for a --fail flag to demonstrate non-zero exit codes */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--fail") == 0) {
            fprintf(stderr, "Error: --fail flag was passed. Exiting with failure.\n");
            return EXIT_FAILURE;  /* returns 1 to the shell */
        }
    }

    print_build_info();
    demonstrate_arguments(argc, argv);
    demonstrate_env();
    demonstrate_exit_codes();
    demonstrate_ub_concept();

    printf("=== Done ===\n");
    printf("Program exiting with EXIT_SUCCESS (%d).\n", EXIT_SUCCESS);

    return EXIT_SUCCESS;  /* 0 — tells the shell "everything went fine" */
}
