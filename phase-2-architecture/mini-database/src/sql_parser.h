#ifndef SQL_PARSER_H
#define SQL_PARSER_H

/*
 * sql_parser.h — minimal SQL parser for the three supported statements:
 *
 *   INSERT INTO rows VALUES (id, 'name', score);
 *   SELECT * FROM rows;
 *   SELECT * FROM rows WHERE id = N;
 *
 * The parser is case-insensitive for keywords.
 * Name must be a single-quoted string.
 * id is a positive integer, score is a floating-point number.
 *
 * Returns a SqlStatement describing what was parsed.
 */

typedef enum {
    SQL_INSERT,        /* INSERT INTO rows VALUES (...) */
    SQL_SELECT_ALL,    /* SELECT * FROM rows            */
    SQL_SELECT_WHERE,  /* SELECT * FROM rows WHERE id=N */
    SQL_UNKNOWN,       /* unrecognised or parse error   */
    SQL_EXIT           /* special: user typed "exit"    */
} SqlType;

typedef struct {
    SqlType type;

    /* For INSERT */
    int    insert_id;
    char   insert_name[64];
    double insert_score;

    /* For SELECT WHERE */
    int    where_id;

    /* Error message (if type == SQL_UNKNOWN) */
    char   error[128];
} SqlStatement;

/* Parse one line of SQL.  Returns a SqlStatement. */
SqlStatement sql_parse(const char *line);

#endif /* SQL_PARSER_H */
