/*
 * codegen.c — x86-64 AT&T assembly code generator for Calc.
 *
 * Two-pass design:
 *   Pass 1: scan_vars() — walk the AST to collect all variable declarations
 *           so we know the total frame size before emitting the prologue.
 *   Pass 2: emit_node() — recursively emit instructions for each AST node.
 *
 * Expression evaluation convention:
 *   Every call to emit_expr(node) leaves the result in %rax.
 *   Binary operations:
 *     1. emit_expr(left)  → result in %rax
 *     2. pushq %rax       → save left on stack
 *     3. emit_expr(right) → result in %rax
 *     4. popq %rcx        → restore left into %rcx
 *     5. compute %rcx op %rax → result in %rax
 *
 *   Note: for non-commutative ops (sub, div, mod, cmp) the order matters.
 *   After popq %rcx: %rcx = left operand, %rax = right operand.
 *   For subtraction: we want left - right, so: subq %rax, %rcx; movq %rcx, %rax
 *   For division: idivq divides %rdx:%rax by the operand.
 *     We want left / right, so: movq %rcx, %rax (left is dividend),
 *     sign-extend into %rdx, idivq %rdi (right, moved to %rdi to free %rax).
 */

#include "codegen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------
 * Initialisation
 * --------------------------------------------------------------- */
void codegen_init(CodeGen *cg, FILE *out)
{
    memset(cg, 0, sizeof(*cg));
    cg->out = out;
}

/* ---------------------------------------------------------------
 * Symbol table helpers
 * --------------------------------------------------------------- */
static int var_offset(CodeGen *cg, const char *name)
{
    for (int i = 0; i < cg->var_count; ++i)
        if (strcmp(cg->vars[i], name) == 0)
            return cg->var_offsets[i];
    fprintf(stderr, "Codegen error: undefined variable '%s'\n", name);
    exit(EXIT_FAILURE);
}

static void var_declare(CodeGen *cg, const char *name)
{
    /* Check for redeclaration */
    for (int i = 0; i < cg->var_count; ++i)
        if (strcmp(cg->vars[i], name) == 0) return;  /* already declared */
    if (cg->var_count >= 64) {
        fprintf(stderr, "Too many variables (max 64)\n");
        exit(EXIT_FAILURE);
    }
    int idx = cg->var_count++;
    strncpy(cg->vars[idx], name, 63);
    cg->var_offsets[idx] = -8 * (idx + 1);  /* -8, -16, -24, … */
}

/* ---------------------------------------------------------------
 * Pass 1: collect all variable declarations
 * --------------------------------------------------------------- */
static void scan_vars(CodeGen *cg, AstNode *node)
{
    if (!node) return;
    switch (node->type) {
    case NODE_PROGRAM:
    case NODE_BLOCK:
        for (int i = 0; i < node->block.count; ++i)
            scan_vars(cg, node->block.stmts[i]);
        break;
    case NODE_VAR_DECL:
        var_declare(cg, node->var_decl.name);
        scan_vars(cg, node->var_decl.init);
        break;
    case NODE_ASSIGN:
        scan_vars(cg, node->assign.value);
        break;
    case NODE_PRINT:
        scan_vars(cg, node->print_expr);
        break;
    case NODE_IF:
        scan_vars(cg, node->if_stmt.cond);
        scan_vars(cg, node->if_stmt.then_body);
        scan_vars(cg, node->if_stmt.else_body);
        break;
    case NODE_WHILE:
        scan_vars(cg, node->while_stmt.cond);
        scan_vars(cg, node->while_stmt.body);
        break;
    case NODE_BINOP:
        scan_vars(cg, node->binop.left);
        scan_vars(cg, node->binop.right);
        break;
    case NODE_UNARY:
        scan_vars(cg, node->unary.operand);
        break;
    case NODE_NUMBER:
    case NODE_IDENT:
        break;
    }
}

/* ---------------------------------------------------------------
 * Label generation
 * --------------------------------------------------------------- */
static int new_label(CodeGen *cg)
{
    return cg->label_count++;
}

#define EMIT(cg, fmt, ...) fprintf((cg)->out, fmt "\n", ##__VA_ARGS__)

/* ---------------------------------------------------------------
 * Pass 2: emit assembly
 * --------------------------------------------------------------- */
static void emit_expr(CodeGen *cg, AstNode *node);
static void emit_stmt(CodeGen *cg, AstNode *node);

static void emit_expr(CodeGen *cg, AstNode *node)
{
    if (!node) return;

    switch (node->type) {
    case NODE_NUMBER:
        EMIT(cg, "    movq    $%ld, %%rax", node->number_val);
        break;

    case NODE_IDENT: {
        int off = var_offset(cg, node->ident_name);
        EMIT(cg, "    movq    %d(%%rbp), %%rax", off);
        break;
    }

    case NODE_UNARY:
        emit_expr(cg, node->unary.operand);
        if (strcmp(node->unary.op, "-") == 0) {
            EMIT(cg, "    negq    %%rax");
        } else if (strcmp(node->unary.op, "!") == 0) {
            /* !x: result is 1 if x==0, else 0 */
            EMIT(cg, "    testq   %%rax, %%rax");
            EMIT(cg, "    sete    %%al");
            EMIT(cg, "    movzbq  %%al, %%rax");
        }
        break;

    case NODE_BINOP: {
        const char *op = node->binop.op;

        /* Short-circuit evaluation for && and || */
        if (strcmp(op, "&&") == 0) {
            int lbl_false = new_label(cg);
            int lbl_end   = new_label(cg);
            emit_expr(cg, node->binop.left);
            EMIT(cg, "    testq   %%rax, %%rax");
            EMIT(cg, "    je      .L%d", lbl_false);
            emit_expr(cg, node->binop.right);
            EMIT(cg, "    testq   %%rax, %%rax");
            EMIT(cg, "    je      .L%d", lbl_false);
            EMIT(cg, "    movq    $1, %%rax");
            EMIT(cg, "    jmp     .L%d", lbl_end);
            EMIT(cg, ".L%d:", lbl_false);
            EMIT(cg, "    movq    $0, %%rax");
            EMIT(cg, ".L%d:", lbl_end);
            break;
        }
        if (strcmp(op, "||") == 0) {
            int lbl_true = new_label(cg);
            int lbl_end  = new_label(cg);
            emit_expr(cg, node->binop.left);
            EMIT(cg, "    testq   %%rax, %%rax");
            EMIT(cg, "    jne     .L%d", lbl_true);
            emit_expr(cg, node->binop.right);
            EMIT(cg, "    testq   %%rax, %%rax");
            EMIT(cg, "    jne     .L%d", lbl_true);
            EMIT(cg, "    movq    $0, %%rax");
            EMIT(cg, "    jmp     .L%d", lbl_end);
            EMIT(cg, ".L%d:", lbl_true);
            EMIT(cg, "    movq    $1, %%rax");
            EMIT(cg, ".L%d:", lbl_end);
            break;
        }

        /* Standard: left into %rcx, right into %rax, then operate */
        emit_expr(cg, node->binop.left);
        EMIT(cg, "    pushq   %%rax");           /* save left  */
        emit_expr(cg, node->binop.right);        /* rax = right */
        EMIT(cg, "    popq    %%rcx");            /* rcx = left  */

        if (strcmp(op, "+") == 0) {
            EMIT(cg, "    addq    %%rcx, %%rax");
        } else if (strcmp(op, "-") == 0) {
            /* left - right: rcx - rax */
            EMIT(cg, "    subq    %%rax, %%rcx");
            EMIT(cg, "    movq    %%rcx, %%rax");
        } else if (strcmp(op, "*") == 0) {
            EMIT(cg, "    imulq   %%rcx, %%rax");
        } else if (strcmp(op, "/") == 0 || strcmp(op, "%") == 0) {
            /* idivq: divides %rdx:%rax by operand.
             * We want left / right: dividend = rcx (left), divisor = rax (right).
             * Move right to %rdi (caller-saved), set dividend. */
            EMIT(cg, "    movq    %%rax, %%rdi");   /* rdi = right (divisor)  */
            EMIT(cg, "    movq    %%rcx, %%rax");   /* rax = left  (dividend) */
            EMIT(cg, "    cqto");                   /* sign-extend rax → rdx:rax */
            EMIT(cg, "    idivq   %%rdi");           /* rax = quotient, rdx = remainder */
            if (strcmp(op, "%") == 0)
                EMIT(cg, "    movq    %%rdx, %%rax"); /* return remainder */
        } else if (strcmp(op, "==") == 0) {
            EMIT(cg, "    cmpq    %%rax, %%rcx");
            EMIT(cg, "    sete    %%al");
            EMIT(cg, "    movzbq  %%al, %%rax");
        } else if (strcmp(op, "!=") == 0) {
            EMIT(cg, "    cmpq    %%rax, %%rcx");
            EMIT(cg, "    setne   %%al");
            EMIT(cg, "    movzbq  %%al, %%rax");
        } else if (strcmp(op, "<") == 0) {
            EMIT(cg, "    cmpq    %%rax, %%rcx");
            EMIT(cg, "    setl    %%al");
            EMIT(cg, "    movzbq  %%al, %%rax");
        } else if (strcmp(op, ">") == 0) {
            EMIT(cg, "    cmpq    %%rax, %%rcx");
            EMIT(cg, "    setg    %%al");
            EMIT(cg, "    movzbq  %%al, %%rax");
        } else if (strcmp(op, "<=") == 0) {
            EMIT(cg, "    cmpq    %%rax, %%rcx");
            EMIT(cg, "    setle   %%al");
            EMIT(cg, "    movzbq  %%al, %%rax");
        } else if (strcmp(op, ">=") == 0) {
            EMIT(cg, "    cmpq    %%rax, %%rcx");
            EMIT(cg, "    setge   %%al");
            EMIT(cg, "    movzbq  %%al, %%rax");
        } else {
            fprintf(stderr, "Codegen: unknown operator '%s'\n", op);
            exit(EXIT_FAILURE);
        }
        break;
    }

    default:
        fprintf(stderr, "Codegen: unexpected node type %d in expression\n", node->type);
        exit(EXIT_FAILURE);
    }
}

static void emit_stmt(CodeGen *cg, AstNode *node)
{
    if (!node) return;

    switch (node->type) {
    case NODE_PROGRAM:
    case NODE_BLOCK:
        for (int i = 0; i < node->block.count; ++i)
            emit_stmt(cg, node->block.stmts[i]);
        break;

    case NODE_VAR_DECL: {
        /* Evaluate init expression and store to stack slot */
        emit_expr(cg, node->var_decl.init);
        int off = var_offset(cg, node->var_decl.name);
        EMIT(cg, "    movq    %%rax, %d(%%rbp)   # var %s", off, node->var_decl.name);
        break;
    }

    case NODE_ASSIGN: {
        emit_expr(cg, node->assign.value);
        int off = var_offset(cg, node->assign.name);
        EMIT(cg, "    movq    %%rax, %d(%%rbp)   # %s =", off, node->assign.name);
        break;
    }

    case NODE_PRINT: {
        /*
         * Call printf("%ld\n", value) using System V AMD64 ABI:
         *   %rdi = first arg  (format string pointer)
         *   %rsi = second arg (value)
         *   %eax = 0          (no XMM registers used)
         */
        emit_expr(cg, node->print_expr);
        EMIT(cg, "    movq    %%rax, %%rsi");
        EMIT(cg, "    leaq    .fmt_int(%%rip), %%rdi");
        EMIT(cg, "    xorl    %%eax, %%eax");
#ifdef __APPLE__
        EMIT(cg, "    call    _printf");
#else
        EMIT(cg, "    call    printf");
#endif
        break;
    }

    case NODE_IF: {
        int lbl_else = new_label(cg);
        int lbl_end  = new_label(cg);
        emit_expr(cg, node->if_stmt.cond);
        EMIT(cg, "    testq   %%rax, %%rax");
        if (node->if_stmt.else_body)
            EMIT(cg, "    je      .L%d", lbl_else);
        else
            EMIT(cg, "    je      .L%d", lbl_end);
        emit_stmt(cg, node->if_stmt.then_body);
        if (node->if_stmt.else_body) {
            EMIT(cg, "    jmp     .L%d", lbl_end);
            EMIT(cg, ".L%d:", lbl_else);
            emit_stmt(cg, node->if_stmt.else_body);
        }
        EMIT(cg, ".L%d:", lbl_end);
        break;
    }

    case NODE_WHILE: {
        int lbl_loop = new_label(cg);
        int lbl_end  = new_label(cg);
        EMIT(cg, ".L%d:", lbl_loop);
        emit_expr(cg, node->while_stmt.cond);
        EMIT(cg, "    testq   %%rax, %%rax");
        EMIT(cg, "    je      .L%d", lbl_end);
        emit_stmt(cg, node->while_stmt.body);
        EMIT(cg, "    jmp     .L%d", lbl_loop);
        EMIT(cg, ".L%d:", lbl_end);
        break;
    }

    default:
        fprintf(stderr, "Codegen: unexpected node type %d in statement\n", node->type);
        exit(EXIT_FAILURE);
    }
}

/* ---------------------------------------------------------------
 * Public: generate complete .s file
 * --------------------------------------------------------------- */
void codegen_program(CodeGen *cg, AstNode *program)
{
    /* Pass 1: collect variables */
    scan_vars(cg, program);

    /* Frame size: 8 bytes per variable, rounded up to 16-byte alignment */
    int frame_size = cg->var_count * 8;
    if (frame_size % 16 != 0)
        frame_size = (frame_size + 15) & ~15;
    if (frame_size == 0)
        frame_size = 16;  /* always reserve at least 16 bytes */

    /* --- Data section: format string for printf ------------------- */
#ifdef __APPLE__
    EMIT(cg, "    .section __TEXT,__cstring,cstring_literals");
#else
    EMIT(cg, "    .section .rodata");
#endif
    EMIT(cg, ".fmt_int:");
    EMIT(cg, "    .string \"%%ld\\n\"");
    EMIT(cg, "");

    /* --- Text section + function prologue ------------------------- */
    EMIT(cg, "    .section .text");
    EMIT(cg, "    .global main");
#ifndef __APPLE__
    EMIT(cg, "    .type   main, @function");
#endif
    EMIT(cg, "main:");
    EMIT(cg, "    pushq   %%rbp");
    EMIT(cg, "    movq    %%rsp, %%rbp");
    EMIT(cg, "    subq    $%d, %%rsp", frame_size);
    EMIT(cg, "");

    /* --- Pass 2: emit body ---------------------------------------- */
    emit_stmt(cg, program);

    /* --- Function epilogue ---------------------------------------- */
    EMIT(cg, "");
    EMIT(cg, "    movq    $0, %%rax");
    EMIT(cg, "    leave");
    EMIT(cg, "    ret");
    EMIT(cg, "");
#ifndef __APPLE__
    EMIT(cg, "    .size main, .-main");
#endif
}
