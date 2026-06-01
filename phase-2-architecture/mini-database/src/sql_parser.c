/*
 * sql_parser.c — minimal SQL parser.
 *
 * Implementation: simple hand-written parser using strtok-style tokenisation.
 * We use sscanf for structured value extraction after we've identified the
 * statement type.
 *
 * No full grammar.  Pattern matching on the first keyword suffices given
 * our limited SQL dialect.
 */

#include "sql_parser.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Copy src to dest and convert to lower-case, up to max-1 chars. */
static void to_lower(const char *src, char *dest, size_t max)
{
    size_t i;
    for (i = 0; i + 1 < max && src[i]; ++i)
        dest[i] = (char)tolower((unsigned char)src[i]);
    dest[i] = '\0';
}

/* Trim leading and trailing whitespace in-place. */
static char *trim(char *s)
{
    while (*s && isspace((unsigned char)*s)) ++s;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)*(end-1))) --end;
    *end = '\0';
    return s;
}

SqlStatement sql_parse(const char *raw_line)
{
    SqlStatement stmt;
    memset(&stmt, 0, sizeof(stmt));
    stmt.type = SQL_UNKNOWN;

    /* Work on a mutable copy */
    char line[512];
    strncpy(line, raw_line, sizeof(line) - 1);
    line[sizeof(line)-1] = '\0';
    char *s = trim(line);

    if (*s == '\0') {
        /* empty line */
        stmt.type = SQL_UNKNOWN;
        strcpy(stmt.error, "empty input");
        return stmt;
    }

    /* Lowercase copy for keyword detection */
    char lower[512];
    to_lower(s, lower, sizeof(lower));

    /* --- exit ----------------------------------------------- */
    if (strcmp(lower, "exit") == 0 || strcmp(lower, "quit") == 0) {
        stmt.type = SQL_EXIT;
        return stmt;
    }

    /* --- INSERT INTO rows VALUES (id, 'name', score) -------- */
    if (strncmp(lower, "insert", 6) == 0) {
        /* Extract the part inside parentheses */
        char *lparen = strchr(s, '(');
        char *rparen = strrchr(s, ')');
        if (!lparen || !rparen || rparen <= lparen) {
            snprintf(stmt.error, sizeof(stmt.error),
                     "Expected VALUES (id, 'name', score)");
            return stmt;
        }

        /* Content between parens */
        char values[256];
        size_t vlen = (size_t)(rparen - lparen - 1);
        if (vlen >= sizeof(values)) vlen = sizeof(values) - 1;
        strncpy(values, lparen + 1, vlen);
        values[vlen] = '\0';

        /* Parse: id, 'name', score
         * Find the first comma (separates id from name) */
        char *p = values;
        while (*p && isspace((unsigned char)*p)) ++p;

        /* id */
        char *end_id;
        long id = strtol(p, &end_id, 10);
        if (end_id == p) {
            snprintf(stmt.error, sizeof(stmt.error), "Expected integer id");
            return stmt;
        }
        p = end_id;
        while (*p && (isspace((unsigned char)*p) || *p == ',')) ++p;

        /* 'name' — find single-quoted string */
        if (*p != '\'') {
            snprintf(stmt.error, sizeof(stmt.error),
                     "Expected single-quoted name, got '%c'", *p);
            return stmt;
        }
        ++p;  /* skip opening quote */
        char *name_start = p;
        while (*p && *p != '\'') ++p;
        if (*p != '\'') {
            snprintf(stmt.error, sizeof(stmt.error), "Unterminated string");
            return stmt;
        }
        size_t nlen = (size_t)(p - name_start);
        if (nlen >= sizeof(stmt.insert_name)) nlen = sizeof(stmt.insert_name) - 1;
        strncpy(stmt.insert_name, name_start, nlen);
        stmt.insert_name[nlen] = '\0';
        ++p;  /* skip closing quote */
        while (*p && (isspace((unsigned char)*p) || *p == ',')) ++p;

        /* score */
        char *end_score;
        double score = strtod(p, &end_score);
        if (end_score == p) {
            snprintf(stmt.error, sizeof(stmt.error), "Expected numeric score");
            return stmt;
        }

        stmt.type        = SQL_INSERT;
        stmt.insert_id   = (int)id;
        stmt.insert_score = score;
        return stmt;
    }

    /* --- SELECT * FROM rows [WHERE id = N] ------------------- */
    if (strncmp(lower, "select", 6) == 0) {
        /* Check for WHERE clause */
        char *where = strstr(lower, "where");
        if (!where) {
            stmt.type = SQL_SELECT_ALL;
            return stmt;
        }

        /* Parse "where id = N" */
        char *after_where = where + 5;
        while (*after_where && isspace((unsigned char)*after_where)) ++after_where;

        /* skip "id" */
        if (strncmp(after_where, "id", 2) != 0) {
            snprintf(stmt.error, sizeof(stmt.error),
                     "Only WHERE id = N is supported");
            return stmt;
        }
        after_where += 2;
        while (*after_where && isspace((unsigned char)*after_where)) ++after_where;
        if (*after_where != '=') {
            snprintf(stmt.error, sizeof(stmt.error), "Expected '=' after id");
            return stmt;
        }
        ++after_where;
        while (*after_where && isspace((unsigned char)*after_where)) ++after_where;

        /* The number is in the lower-case copy; get position and use
         * the same offset in the original line for the digits. */
        char *end_n;
        long n = strtol(after_where, &end_n, 10);
        if (end_n == after_where) {
            snprintf(stmt.error, sizeof(stmt.error), "Expected integer after '='");
            return stmt;
        }

        stmt.type     = SQL_SELECT_WHERE;
        stmt.where_id = (int)n;
        return stmt;
    }

    snprintf(stmt.error, sizeof(stmt.error),
             "Unknown statement: %.40s", s);
    return stmt;
}
