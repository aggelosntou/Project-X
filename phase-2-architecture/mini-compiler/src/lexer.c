/*
 * lexer.c — tokeniser for the Calc language.
 *
 * Design:
 *   - Single-pass over the source string using a cursor (pos index).
 *   - lexer_peek() saves/restores the cursor to implement non-consuming look-ahead.
 *   - Keywords are identified by comparing identifier text against a table.
 *   - Single-line comments (//) are silently skipped.
 */

#include "lexer.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------
 * Initialisation
 * --------------------------------------------------------------- */
void lexer_init(Lexer *l, const char *src)
{
    l->src  = src;
    l->pos  = 0;
    l->line = 1;
}

/* ---------------------------------------------------------------
 * Internal helpers
 * --------------------------------------------------------------- */
static char peek_char(const Lexer *l)
{
    return l->src[l->pos];
}

static char next_char(Lexer *l)
{
    char c = l->src[l->pos++];
    if (c == '\n') l->line++;
    return c;
}

static void skip_whitespace_and_comments(Lexer *l)
{
    for (;;) {
        /* Skip whitespace */
        while (isspace((unsigned char)peek_char(l)))
            next_char(l);

        /* Skip // line comment */
        if (peek_char(l) == '/' && l->src[l->pos + 1] == '/') {
            while (peek_char(l) != '\0' && peek_char(l) != '\n')
                next_char(l);
            continue;
        }
        break;
    }
}

/* Map identifier text to keyword token type, or TOK_IDENT if not a keyword. */
static TokenType keyword_or_ident(const char *text)
{
    if (strcmp(text, "var")   == 0) return TOK_VAR;
    if (strcmp(text, "if")    == 0) return TOK_IF;
    if (strcmp(text, "else")  == 0) return TOK_ELSE;
    if (strcmp(text, "while") == 0) return TOK_WHILE;
    if (strcmp(text, "print") == 0) return TOK_PRINT;
    return TOK_IDENT;
}

/* ---------------------------------------------------------------
 * Core: scan and return one token
 * --------------------------------------------------------------- */
static Token scan(Lexer *l)
{
    Token tok;
    tok.ival = 0;
    tok.text[0] = '\0';

    skip_whitespace_and_comments(l);
    tok.line = l->line;

    char c = peek_char(l);

    /* EOF */
    if (c == '\0') {
        tok.type = TOK_EOF;
        strcpy(tok.text, "<EOF>");
        return tok;
    }

    /* Integer literal */
    if (isdigit((unsigned char)c)) {
        int start = l->pos;
        while (isdigit((unsigned char)peek_char(l)))
            next_char(l);
        int len = l->pos - start;
        if (len >= 63) len = 63;
        memcpy(tok.text, l->src + start, len);
        tok.text[len] = '\0';
        tok.ival = atol(tok.text);
        tok.type = TOK_NUMBER;
        return tok;
    }

    /* Identifier or keyword */
    if (isalpha((unsigned char)c) || c == '_') {
        int start = l->pos;
        while (isalnum((unsigned char)peek_char(l)) || peek_char(l) == '_')
            next_char(l);
        int len = l->pos - start;
        if (len >= 63) len = 63;
        memcpy(tok.text, l->src + start, len);
        tok.text[len] = '\0';
        tok.type = keyword_or_ident(tok.text);
        return tok;
    }

    /* Consume the character */
    next_char(l);

    /* Two-character operators */
    char c2 = peek_char(l);

    switch (c) {
    case '=':
        if (c2 == '=') { next_char(l); tok.type = TOK_EQEQ; strcpy(tok.text, "=="); return tok; }
        tok.type = TOK_EQ; strcpy(tok.text, "="); return tok;

    case '!':
        if (c2 == '=') { next_char(l); tok.type = TOK_NEQ; strcpy(tok.text, "!="); return tok; }
        tok.type = TOK_NOT; strcpy(tok.text, "!"); return tok;

    case '<':
        if (c2 == '=') { next_char(l); tok.type = TOK_LEQ; strcpy(tok.text, "<="); return tok; }
        tok.type = TOK_LT; strcpy(tok.text, "<"); return tok;

    case '>':
        if (c2 == '=') { next_char(l); tok.type = TOK_GEQ; strcpy(tok.text, ">="); return tok; }
        tok.type = TOK_GT; strcpy(tok.text, ">"); return tok;

    case '&':
        if (c2 == '&') { next_char(l); tok.type = TOK_AND; strcpy(tok.text, "&&"); return tok; }
        goto error;

    case '|':
        if (c2 == '|') { next_char(l); tok.type = TOK_OR; strcpy(tok.text, "||"); return tok; }
        goto error;

    /* Single-character tokens */
    case '+': tok.type = TOK_PLUS;    strcpy(tok.text, "+"); return tok;
    case '-': tok.type = TOK_MINUS;   strcpy(tok.text, "-"); return tok;
    case '*': tok.type = TOK_STAR;    strcpy(tok.text, "*"); return tok;
    case '/': tok.type = TOK_SLASH;   strcpy(tok.text, "/"); return tok;
    case '%': tok.type = TOK_PERCENT; strcpy(tok.text, "%"); return tok;
    case '(': tok.type = TOK_LPAREN;  strcpy(tok.text, "("); return tok;
    case ')': tok.type = TOK_RPAREN;  strcpy(tok.text, ")"); return tok;
    case '{': tok.type = TOK_LBRACE;  strcpy(tok.text, "{"); return tok;
    case '}': tok.type = TOK_RBRACE;  strcpy(tok.text, "}"); return tok;
    case ';': tok.type = TOK_SEMI;    strcpy(tok.text, ";"); return tok;

    default:
        goto error;
    }

error:
    tok.type = TOK_ERROR;
    tok.text[0] = c;
    tok.text[1] = '\0';
    fprintf(stderr, "Lexer error at line %d: unexpected character '%c'\n",
            tok.line, c);
    return tok;
}

/* ---------------------------------------------------------------
 * Public API
 * --------------------------------------------------------------- */
Token lexer_next(Lexer *l)
{
    return scan(l);
}

Token lexer_peek(Lexer *l)
{
    /* Save state, scan one token, restore state. */
    int saved_pos  = l->pos;
    int saved_line = l->line;
    Token t = scan(l);
    l->pos  = saved_pos;
    l->line = saved_line;
    return t;
}

const char *token_type_name(TokenType t)
{
    switch (t) {
    case TOK_NUMBER:  return "NUMBER";
    case TOK_IDENT:   return "IDENT";
    case TOK_PLUS:    return "+";
    case TOK_MINUS:   return "-";
    case TOK_STAR:    return "*";
    case TOK_SLASH:   return "/";
    case TOK_PERCENT: return "%";
    case TOK_EQ:      return "=";
    case TOK_EQEQ:    return "==";
    case TOK_NEQ:     return "!=";
    case TOK_LT:      return "<";
    case TOK_GT:      return ">";
    case TOK_LEQ:     return "<=";
    case TOK_GEQ:     return ">=";
    case TOK_AND:     return "&&";
    case TOK_OR:      return "||";
    case TOK_NOT:     return "!";
    case TOK_LPAREN:  return "(";
    case TOK_RPAREN:  return ")";
    case TOK_LBRACE:  return "{";
    case TOK_RBRACE:  return "}";
    case TOK_SEMI:    return ";";
    case TOK_VAR:     return "var";
    case TOK_IF:      return "if";
    case TOK_ELSE:    return "else";
    case TOK_WHILE:   return "while";
    case TOK_PRINT:   return "print";
    case TOK_EOF:     return "EOF";
    case TOK_ERROR:   return "ERROR";
    }
    return "?";
}
