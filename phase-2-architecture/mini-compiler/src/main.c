/*
 * main.c — driver for the Calc compiler.
 *
 * Usage:
 *   ./calc <source.calc>
 *
 * Steps:
 *   1. Read the entire .calc source file into memory.
 *   2. Lex → Parse → produce AST.
 *   3. Code-generate → write .s assembly file.
 *   4. Optionally assemble and link with gcc.
 *
 * Output filename: derived from the input filename by replacing .calc with .s
 *   e.g.  programs/add.calc → add.s
 *         fizzbuzz.calc     → fizzbuzz.s
 *
 * The -S flag (stop after generating .s) is always the default.
 * Use  gcc -no-pie <output>.s -o <output>  to produce an executable.
 * On macOS:  gcc <output>.s -o <output>   (no-pie may not be needed)
 */

#include "parser.h"
#include "codegen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------
 * Read the entire contents of a file into a heap-allocated string.
 * Caller must free the returned pointer.
 * --------------------------------------------------------------- */
static char *read_file(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Cannot open '%s': ", path);
        perror("");
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);

    char *buf = (char *)malloc(size + 1);
    if (!buf) { fclose(f); perror("malloc"); return NULL; }

    size_t nread = fread(buf, 1, size, f);
    buf[nread] = '\0';
    fclose(f);
    return buf;
}

/* ---------------------------------------------------------------
 * Derive output filename from input filename.
 * "programs/add.calc" → "add.s"
 * "loop.calc"         → "loop.s"
 * --------------------------------------------------------------- */
static void make_output_name(const char *input, char *out, size_t out_size)
{
    /* Strip directory prefix */
    const char *base = strrchr(input, '/');
    base = base ? base + 1 : input;

    /* Copy base name */
    strncpy(out, base, out_size - 1);
    out[out_size - 1] = '\0';

    /* Replace .calc extension with .s, or append .s */
    char *ext = strrchr(out, '.');
    if (ext && strcmp(ext, ".calc") == 0) {
        strcpy(ext, ".s");
    } else {
        size_t len = strlen(out);
        if (len + 2 < out_size) {
            out[len]     = '.';
            out[len + 1] = 's';
            out[len + 2] = '\0';
        }
    }
}

/* ---------------------------------------------------------------
 * Main
 * --------------------------------------------------------------- */
int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <source.calc>\n", argv[0]);
        fprintf(stderr, "  Compiles a Calc source file to x86-64 AT&T assembly.\n");
        return EXIT_FAILURE;
    }

    const char *input_path = argv[1];

    /* --- 1. Read source --------------------------------------- */
    char *src = read_file(input_path);
    if (!src) return EXIT_FAILURE;

    /* --- 2. Parse -------------------------------------------- */
    AstNode *ast = parse(src);
    free(src);
    if (!ast) {
        fprintf(stderr, "Compilation failed: parse error.\n");
        return EXIT_FAILURE;
    }

    /* Print AST for educational purposes when CALC_DUMP_AST=1 */
    if (getenv("CALC_DUMP_AST")) {
        printf("=== AST ===\n");
        ast_print(ast, 0);
        printf("===========\n\n");
    }

    /* --- 3. Open output file --------------------------------- */
    char out_name[256];
    make_output_name(input_path, out_name, sizeof(out_name));

    FILE *out = fopen(out_name, "w");
    if (!out) {
        fprintf(stderr, "Cannot create '%s': ", out_name);
        perror("");
        ast_free(ast);
        return EXIT_FAILURE;
    }

    /* --- 4. Code generation ---------------------------------- */
    CodeGen cg;
    codegen_init(&cg, out);
    codegen_program(&cg, ast);
    fclose(out);
    ast_free(ast);

    printf("Compiled '%s' → '%s'\n", input_path, out_name);
    printf("To build an executable:\n");

#ifdef __APPLE__
    /* macOS: clang/gcc does not need -no-pie */
    char base[256];
    strncpy(base, out_name, sizeof(base) - 1);
    char *dot = strrchr(base, '.');
    if (dot) *dot = '\0';
    printf("  gcc %s -o %s\n", out_name, base);
    printf("  ./%s\n", base);
#else
    char base[256];
    strncpy(base, out_name, sizeof(base) - 1);
    char *dot = strrchr(base, '.');
    if (dot) *dot = '\0';
    printf("  gcc -no-pie %s -o %s\n", out_name, base);
    printf("  ./%s\n", base);
#endif

    return EXIT_SUCCESS;
}
