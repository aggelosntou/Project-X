/*
 * parser.c — recursive-descent parser for Calc.
 *
 * Each grammar rule maps to one C function.  The parser consumes tokens
 * from the Lexer one at a time using expect() / accept() helpers.
 *
 * Error recovery: on any parse error we print a message and return NULL,
 * which propagates up to the caller.
 */

#include "parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------
 * Parser state (kept as a local struct, passed by pointer)
 * --------------------------------------------------------------- */
typedef struct {
    Lexer lexer;
    Token current;   /* the token most recently consumed */
    Token lookahead; /* one token of look-ahead           */
    int   had_error;
} Parser;

/* ---------------------------------------------------------------
 * Node allocation helpers
 * --------------------------------------------------------------- */
static AstNode *node_new(NodeType type, int line)
{
    AstNode *n = (AstNode *)calloc(1, sizeof(AstNode));
    if (!n) { perror("calloc"); exit(EXIT_FAILURE); }
    n->type = type;
    n->line = line;
    return n;
}

static void block_append(AstNode *block_node, AstNode *stmt)
{
    if (block_node->block.count == block_node->block.capacity) {
        int new_cap = block_node->block.capacity ? block_node->block.capacity * 2 : 8;
        block_node->block.stmts = (AstNode **)realloc(
            block_node->block.stmts, new_cap * sizeof(AstNode *));
        if (!block_node->block.stmts) { perror("realloc"); exit(EXIT_FAILURE); }
        block_node->block.capacity = new_cap;
    }
    block_node->block.stmts[block_node->block.count++] = stmt;
}

/* ---------------------------------------------------------------
 * Token helpers
 * --------------------------------------------------------------- */

/* Advance past lookahead, making it the current token. */
static Token advance(Parser *p)
{
    p->current   = p->lookahead;
    p->lookahead = lexer_next(&p->lexer);
    return p->current;
}

/* Check if lookahead has the given type without consuming it. */
static int check(Parser *p, TokenType t)
{
    return p->lookahead.type == t;
}

/* Consume lookahead if it matches t; return 1 on success, 0 otherwise. */
static int accept(Parser *p, TokenType t)
{
    if (check(p, t)) { advance(p); return 1; }
    return 0;
}

/* Consume lookahead if it matches t; print error and set had_error on failure. */
static int expect(Parser *p, TokenType t)
{
    if (accept(p, t)) return 1;
    fprintf(stderr, "Parse error at line %d: expected '%s', got '%s' ('%s')\n",
            p->lookahead.line,
            token_type_name(t),
            token_type_name(p->lookahead.type),
            p->lookahead.text);
    p->had_error = 1;
    return 0;
}

/* ---------------------------------------------------------------
 * Forward declarations
 * --------------------------------------------------------------- */
static AstNode *parse_stmt(Parser *p);
static AstNode *parse_expr(Parser *p);
static AstNode *parse_block(Parser *p);

/* ---------------------------------------------------------------
 * Expression parsing (precedence climbing via separate functions)
 * --------------------------------------------------------------- */

static AstNode *parse_primary(Parser *p)
{
    /* NUMBER */
    if (check(p, TOK_NUMBER)) {
        advance(p);
        AstNode *n = node_new(NODE_NUMBER, p->current.line);
        n->number_val = p->current.ival;
        return n;
    }

    /* IDENT */
    if (check(p, TOK_IDENT)) {
        advance(p);
        AstNode *n = node_new(NODE_IDENT, p->current.line);
        strncpy(n->ident_name, p->current.text, 63);
        n->ident_name[63] = '\0';
        return n;
    }

    /* ( expr ) */
    if (accept(p, TOK_LPAREN)) {
        AstNode *inner = parse_expr(p);
        expect(p, TOK_RPAREN);
        return inner;
    }

    fprintf(stderr, "Parse error at line %d: unexpected token '%s' in expression\n",
            p->lookahead.line, p->lookahead.text);
    p->had_error = 1;
    return NULL;
}

static AstNode *parse_unary(Parser *p)
{
    int line = p->lookahead.line;

    if (check(p, TOK_MINUS) || check(p, TOK_NOT)) {
        advance(p);
        char op[4];
        strncpy(op, p->current.text, 3);
        op[3] = '\0';
        AstNode *operand = parse_unary(p);
        if (!operand) return NULL;
        AstNode *n = node_new(NODE_UNARY, line);
        strncpy(n->unary.op, op, 3);
        n->unary.operand = operand;
        return n;
    }
    return parse_primary(p);
}

static AstNode *parse_mul(Parser *p)
{
    AstNode *left = parse_unary(p);
    if (!left) return NULL;

    while (check(p, TOK_STAR) || check(p, TOK_SLASH) || check(p, TOK_PERCENT)) {
        int line = p->lookahead.line;
        advance(p);
        char op[4];
        strncpy(op, p->current.text, 3); op[3] = '\0';
        AstNode *right = parse_unary(p);
        if (!right) return NULL;
        AstNode *n = node_new(NODE_BINOP, line);
        strncpy(n->binop.op, op, 3);
        n->binop.left  = left;
        n->binop.right = right;
        left = n;
    }
    return left;
}

static AstNode *parse_add(Parser *p)
{
    AstNode *left = parse_mul(p);
    if (!left) return NULL;

    while (check(p, TOK_PLUS) || check(p, TOK_MINUS)) {
        int line = p->lookahead.line;
        advance(p);
        char op[4];
        strncpy(op, p->current.text, 3); op[3] = '\0';
        AstNode *right = parse_mul(p);
        if (!right) return NULL;
        AstNode *n = node_new(NODE_BINOP, line);
        strncpy(n->binop.op, op, 3);
        n->binop.left  = left;
        n->binop.right = right;
        left = n;
    }
    return left;
}

static AstNode *parse_rel(Parser *p)
{
    AstNode *left = parse_add(p);
    if (!left) return NULL;

    while (check(p, TOK_LT) || check(p, TOK_GT) ||
           check(p, TOK_LEQ) || check(p, TOK_GEQ)) {
        int line = p->lookahead.line;
        advance(p);
        char op[4];
        strncpy(op, p->current.text, 3); op[3] = '\0';
        AstNode *right = parse_add(p);
        if (!right) return NULL;
        AstNode *n = node_new(NODE_BINOP, line);
        strncpy(n->binop.op, op, 3);
        n->binop.left  = left;
        n->binop.right = right;
        left = n;
    }
    return left;
}

static AstNode *parse_eq(Parser *p)
{
    AstNode *left = parse_rel(p);
    if (!left) return NULL;

    while (check(p, TOK_EQEQ) || check(p, TOK_NEQ)) {
        int line = p->lookahead.line;
        advance(p);
        char op[4];
        strncpy(op, p->current.text, 3); op[3] = '\0';
        AstNode *right = parse_rel(p);
        if (!right) return NULL;
        AstNode *n = node_new(NODE_BINOP, line);
        strncpy(n->binop.op, op, 3);
        n->binop.left  = left;
        n->binop.right = right;
        left = n;
    }
    return left;
}

static AstNode *parse_and(Parser *p)
{
    AstNode *left = parse_eq(p);
    if (!left) return NULL;

    while (check(p, TOK_AND)) {
        int line = p->lookahead.line;
        advance(p);
        AstNode *right = parse_eq(p);
        if (!right) return NULL;
        AstNode *n = node_new(NODE_BINOP, line);
        strcpy(n->binop.op, "&&");
        n->binop.left  = left;
        n->binop.right = right;
        left = n;
    }
    return left;
}

static AstNode *parse_or(Parser *p)
{
    AstNode *left = parse_and(p);
    if (!left) return NULL;

    while (check(p, TOK_OR)) {
        int line = p->lookahead.line;
        advance(p);
        AstNode *right = parse_and(p);
        if (!right) return NULL;
        AstNode *n = node_new(NODE_BINOP, line);
        strcpy(n->binop.op, "||");
        n->binop.left  = left;
        n->binop.right = right;
        left = n;
    }
    return left;
}

static AstNode *parse_expr(Parser *p)
{
    return parse_or(p);
}

/* ---------------------------------------------------------------
 * Statement parsing
 * --------------------------------------------------------------- */

static AstNode *parse_block(Parser *p)
{
    int line = p->lookahead.line;
    if (!expect(p, TOK_LBRACE)) return NULL;

    AstNode *block = node_new(NODE_BLOCK, line);
    while (!check(p, TOK_RBRACE) && !check(p, TOK_EOF)) {
        AstNode *s = parse_stmt(p);
        if (!s || p->had_error) break;
        block_append(block, s);
    }
    expect(p, TOK_RBRACE);
    return block;
}

static AstNode *parse_stmt(Parser *p)
{
    int line = p->lookahead.line;

    /* var name = expr ; */
    if (check(p, TOK_VAR)) {
        advance(p);
        if (!expect(p, TOK_IDENT)) return NULL;
        AstNode *n = node_new(NODE_VAR_DECL, line);
        strncpy(n->var_decl.name, p->current.text, 63);
        if (!expect(p, TOK_EQ)) return NULL;
        n->var_decl.init = parse_expr(p);
        if (!n->var_decl.init) return NULL;
        if (!expect(p, TOK_SEMI)) return NULL;
        return n;
    }

    /* if ( expr ) block [ else block ] */
    if (check(p, TOK_IF)) {
        advance(p);
        AstNode *n = node_new(NODE_IF, line);
        if (!expect(p, TOK_LPAREN)) return NULL;
        n->if_stmt.cond = parse_expr(p);
        if (!n->if_stmt.cond) return NULL;
        if (!expect(p, TOK_RPAREN)) return NULL;
        n->if_stmt.then_body = parse_block(p);
        if (!n->if_stmt.then_body) return NULL;
        if (accept(p, TOK_ELSE)) {
            n->if_stmt.else_body = parse_block(p);
            if (!n->if_stmt.else_body) return NULL;
        }
        return n;
    }

    /* while ( expr ) block */
    if (check(p, TOK_WHILE)) {
        advance(p);
        AstNode *n = node_new(NODE_WHILE, line);
        if (!expect(p, TOK_LPAREN)) return NULL;
        n->while_stmt.cond = parse_expr(p);
        if (!n->while_stmt.cond) return NULL;
        if (!expect(p, TOK_RPAREN)) return NULL;
        n->while_stmt.body = parse_block(p);
        if (!n->while_stmt.body) return NULL;
        return n;
    }

    /* print expr ; */
    if (check(p, TOK_PRINT)) {
        advance(p);
        AstNode *n = node_new(NODE_PRINT, line);
        n->print_expr = parse_expr(p);
        if (!n->print_expr) return NULL;
        if (!expect(p, TOK_SEMI)) return NULL;
        return n;
    }

    /* name = expr ; (assignment) */
    if (check(p, TOK_IDENT)) {
        /* We need to distinguish assignment from a bare expression.
         * In Calc, a statement starting with IDENT must be an assignment. */
        advance(p);   /* consume ident */
        char name[64];
        strncpy(name, p->current.text, 63);
        name[63] = '\0';
        if (!expect(p, TOK_EQ)) return NULL;
        AstNode *n = node_new(NODE_ASSIGN, line);
        strncpy(n->assign.name, name, 63);
        n->assign.value = parse_expr(p);
        if (!n->assign.value) return NULL;
        if (!expect(p, TOK_SEMI)) return NULL;
        return n;
    }

    fprintf(stderr, "Parse error at line %d: unexpected token '%s' at start of statement\n",
            p->lookahead.line, p->lookahead.text);
    p->had_error = 1;
    return NULL;
}

/* ---------------------------------------------------------------
 * Public API
 * --------------------------------------------------------------- */
AstNode *parse(const char *src)
{
    Parser p;
    memset(&p, 0, sizeof(p));
    lexer_init(&p.lexer, src);
    /* Prime the lookahead */
    p.lookahead = lexer_next(&p.lexer);

    AstNode *program = node_new(NODE_PROGRAM, 1);

    while (!check(&p, TOK_EOF) && !p.had_error) {
        AstNode *s = parse_stmt(&p);
        if (!s || p.had_error) break;
        block_append(program, s);
    }

    if (p.had_error) {
        ast_free(program);
        return NULL;
    }
    return program;
}

/* ---------------------------------------------------------------
 * AST printer (for debugging)
 * --------------------------------------------------------------- */
static void indent_print(int indent)
{
    for (int i = 0; i < indent; ++i) printf("  ");
}

void ast_print(AstNode *node, int indent)
{
    if (!node) { indent_print(indent); printf("(null)\n"); return; }

    indent_print(indent);
    switch (node->type) {
    case NODE_PROGRAM:
    case NODE_BLOCK: {
        const char *tag = node->type == NODE_PROGRAM ? "PROGRAM" : "BLOCK";
        printf("%s (%d stmts)\n", tag, node->block.count);
        for (int i = 0; i < node->block.count; ++i)
            ast_print(node->block.stmts[i], indent + 1);
        break;
    }
    case NODE_VAR_DECL:
        printf("VAR_DECL '%s'\n", node->var_decl.name);
        ast_print(node->var_decl.init, indent + 1);
        break;
    case NODE_ASSIGN:
        printf("ASSIGN '%s'\n", node->assign.name);
        ast_print(node->assign.value, indent + 1);
        break;
    case NODE_PRINT:
        printf("PRINT\n");
        ast_print(node->print_expr, indent + 1);
        break;
    case NODE_IF:
        printf("IF\n");
        indent_print(indent + 1); printf("cond:\n");
        ast_print(node->if_stmt.cond, indent + 2);
        indent_print(indent + 1); printf("then:\n");
        ast_print(node->if_stmt.then_body, indent + 2);
        if (node->if_stmt.else_body) {
            indent_print(indent + 1); printf("else:\n");
            ast_print(node->if_stmt.else_body, indent + 2);
        }
        break;
    case NODE_WHILE:
        printf("WHILE\n");
        indent_print(indent + 1); printf("cond:\n");
        ast_print(node->while_stmt.cond, indent + 2);
        indent_print(indent + 1); printf("body:\n");
        ast_print(node->while_stmt.body, indent + 2);
        break;
    case NODE_BINOP:
        printf("BINOP '%s'\n", node->binop.op);
        ast_print(node->binop.left,  indent + 1);
        ast_print(node->binop.right, indent + 1);
        break;
    case NODE_UNARY:
        printf("UNARY '%s'\n", node->unary.op);
        ast_print(node->unary.operand, indent + 1);
        break;
    case NODE_NUMBER:
        printf("NUMBER %ld\n", node->number_val);
        break;
    case NODE_IDENT:
        printf("IDENT '%s'\n", node->ident_name);
        break;
    }
}

/* ---------------------------------------------------------------
 * AST free
 * --------------------------------------------------------------- */
void ast_free(AstNode *node)
{
    if (!node) return;
    switch (node->type) {
    case NODE_PROGRAM:
    case NODE_BLOCK:
        for (int i = 0; i < node->block.count; ++i)
            ast_free(node->block.stmts[i]);
        free(node->block.stmts);
        break;
    case NODE_VAR_DECL: ast_free(node->var_decl.init); break;
    case NODE_ASSIGN:   ast_free(node->assign.value); break;
    case NODE_PRINT:    ast_free(node->print_expr); break;
    case NODE_IF:
        ast_free(node->if_stmt.cond);
        ast_free(node->if_stmt.then_body);
        ast_free(node->if_stmt.else_body);
        break;
    case NODE_WHILE:
        ast_free(node->while_stmt.cond);
        ast_free(node->while_stmt.body);
        break;
    case NODE_BINOP:
        ast_free(node->binop.left);
        ast_free(node->binop.right);
        break;
    case NODE_UNARY:
        ast_free(node->unary.operand);
        break;
    case NODE_NUMBER:
    case NODE_IDENT:
        break;
    }
    free(node);
}
