#ifndef LEXER_H
#define LEXER_H

/*
 * lexer.h — tokeniser for the "Calc" language.
 *
 * Supported tokens:
 *   Numbers      : integer literals (possibly negative after a '-' token)
 *   Identifiers  : [a-zA-Z_][a-zA-Z0-9_]*
 *   Keywords     : var, if, else, while, print
 *   Operators    : + - * / % = == != < > <= >= && || !
 *   Punctuation  : ( ) { } ;
 *   EOF / Error
 */

typedef enum {
    TOK_NUMBER,   /* integer literal                            */
    TOK_IDENT,    /* identifier / keyword text stored in .text */

    /* Arithmetic */
    TOK_PLUS,     /* +  */
    TOK_MINUS,    /* -  */
    TOK_STAR,     /* *  */
    TOK_SLASH,    /* /  */
    TOK_PERCENT,  /* %  */

    /* Assignment and comparison */
    TOK_EQ,       /* =   */
    TOK_EQEQ,     /* ==  */
    TOK_NEQ,      /* !=  */
    TOK_LT,       /* <   */
    TOK_GT,       /* >   */
    TOK_LEQ,      /* <=  */
    TOK_GEQ,      /* >=  */

    /* Logical */
    TOK_AND,      /* &&  */
    TOK_OR,       /* ||  */
    TOK_NOT,      /* !   */

    /* Punctuation */
    TOK_LPAREN,   /* (   */
    TOK_RPAREN,   /* )   */
    TOK_LBRACE,   /* {   */
    TOK_RBRACE,   /* }   */
    TOK_SEMI,     /* ;   */

    /* Keywords (the lexer sets type to these for recognised keywords) */
    TOK_VAR,
    TOK_IF,
    TOK_ELSE,
    TOK_WHILE,
    TOK_PRINT,

    TOK_EOF,
    TOK_ERROR
} TokenType;

typedef struct {
    TokenType type;
    char      text[64];   /* raw text (always NUL-terminated) */
    long      ival;       /* numeric value when type == TOK_NUMBER */
    int       line;       /* 1-based source line for error messages */
} Token;

typedef struct {
    const char *src;   /* full source string (NUL-terminated) */
    int         pos;   /* current read position in src */
    int         line;  /* current line number (starts at 1) */
} Lexer;

/* Initialise a Lexer over the given source string. */
void  lexer_init(Lexer *l, const char *src);

/* Consume and return the next token. */
Token lexer_next(Lexer *l);

/* Return the next token without consuming it. */
Token lexer_peek(Lexer *l);

/* Human-readable name for a token type (useful in error messages). */
const char *token_type_name(TokenType t);

#endif /* LEXER_H */
