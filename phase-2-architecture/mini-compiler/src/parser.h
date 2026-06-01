#ifndef PARSER_H
#define PARSER_H

/*
 * parser.h — AST definition and recursive-descent parser for Calc.
 *
 * Grammar (informal):
 *
 *   program     → stmt*
 *   stmt        → var_decl | assign | if_stmt | while_stmt | print_stmt
 *   var_decl    → 'var' IDENT '=' expr ';'
 *   assign      → IDENT '=' expr ';'
 *   if_stmt     → 'if' '(' expr ')' block ( 'else' block )?
 *   while_stmt  → 'while' '(' expr ')' block
 *   print_stmt  → 'print' expr ';'
 *   block       → '{' stmt* '}'
 *   expr        → or_expr
 *   or_expr     → and_expr  ( '||' and_expr  )*
 *   and_expr    → eq_expr   ( '&&' eq_expr   )*
 *   eq_expr     → rel_expr  ( ('=='|'!=') rel_expr )*
 *   rel_expr    → add_expr  ( ('<'|'>'|'<='|'>=') add_expr )*
 *   add_expr    → mul_expr  ( ('+'|'-') mul_expr  )*
 *   mul_expr    → unary     ( ('*'|'/'|'%') unary   )*
 *   unary       → ('-'|'!') unary | primary
 *   primary     → NUMBER | IDENT | '(' expr ')'
 */

#include "lexer.h"

/* ---------------------------------------------------------------
 * AST node types
 * --------------------------------------------------------------- */
typedef enum {
    NODE_PROGRAM,    /* top-level list of statements                 */
    NODE_BLOCK,      /* { stmt* }                                    */
    NODE_VAR_DECL,   /* var name = init;                             */
    NODE_ASSIGN,     /* name = value;                                */
    NODE_PRINT,      /* print expr;                                  */
    NODE_IF,         /* if (cond) then [else else_body]              */
    NODE_WHILE,      /* while (cond) body                            */
    NODE_BINOP,      /* left op right                                */
    NODE_UNARY,      /* op operand                                   */
    NODE_NUMBER,     /* integer literal                              */
    NODE_IDENT       /* variable reference                           */
} NodeType;

typedef struct AstNode AstNode;

struct AstNode {
    NodeType type;
    int      line;    /* source line for error reporting */

    union {
        /* NODE_NUMBER */
        long number_val;

        /* NODE_IDENT */
        char ident_name[64];

        /* NODE_BINOP */
        struct {
            char     op[4];
            AstNode *left;
            AstNode *right;
        } binop;

        /* NODE_UNARY */
        struct {
            char     op[4];
            AstNode *operand;
        } unary;

        /* NODE_VAR_DECL */
        struct {
            char     name[64];
            AstNode *init;     /* initial value expression */
        } var_decl;

        /* NODE_ASSIGN */
        struct {
            char     name[64];
            AstNode *value;
        } assign;

        /* NODE_PRINT */
        AstNode *print_expr;

        /* NODE_IF */
        struct {
            AstNode *cond;
            AstNode *then_body;
            AstNode *else_body;  /* NULL if no else clause */
        } if_stmt;

        /* NODE_WHILE */
        struct {
            AstNode *cond;
            AstNode *body;
        } while_stmt;

        /* NODE_PROGRAM / NODE_BLOCK */
        struct {
            AstNode **stmts;
            int       count;
            int       capacity;
        } block;
    };
};

/* ---------------------------------------------------------------
 * Public API
 * --------------------------------------------------------------- */

/* Parse the entire source string and return the program AST.
 * Returns NULL on parse error (error message printed to stderr). */
AstNode *parse(const char *src);

/* Pretty-print the AST to stdout for debugging. */
void ast_print(AstNode *node, int indent);

/* Recursively free all AST nodes. */
void ast_free(AstNode *node);

#endif /* PARSER_H */
