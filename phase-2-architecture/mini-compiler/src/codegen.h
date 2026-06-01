#ifndef CODEGEN_H
#define CODEGEN_H

/*
 * codegen.h — x86-64 AT&T assembly code generator for Calc.
 *
 * Target ABI: System V AMD64 (Linux / macOS).
 *
 * Conventions used in generated code:
 *   - %rax  : primary accumulator; every expression leaves its result here.
 *   - %rcx  : temporary for right-hand side of binary operations.
 *   - %rbp  : frame base pointer (callee-saved).
 *   - %rsp  : stack pointer.
 *   - %rdi, %rsi : first and second arguments to called functions (printf).
 *
 * Memory layout:
 *   Every declared variable occupies 8 bytes on the stack relative to %rbp.
 *   var x is the first declared → stored at -8(%rbp)
 *   var y is the second         → stored at -16(%rbp)
 *   …
 *
 * Output format: GNU as / clang assembler (AT&T syntax, .s file).
 */

#include "parser.h"
#include <stdio.h>

typedef struct {
    FILE *out;           /* output assembly file                */
    char  vars[64][64];  /* variable name table                 */
    int   var_offsets[64]; /* stack offset (negative) for each  */
    int   var_count;     /* number of declared variables        */
    int   label_count;   /* counter for generating unique labels*/
} CodeGen;

/* Initialise the code generator, writing assembly to 'out'. */
void codegen_init(CodeGen *cg, FILE *out);

/* Generate a complete .s file from a NODE_PROGRAM AST. */
void codegen_program(CodeGen *cg, AstNode *program);

#endif /* CODEGEN_H */
