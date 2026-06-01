/*
 * main.c — REPL for the mini-database.
 *
 * Usage:
 *   ./minidb [database_file]
 *   (default file: test.db)
 *
 * Supported commands:
 *   INSERT INTO rows VALUES (id, 'name', score);
 *   SELECT * FROM rows;
 *   SELECT * FROM rows WHERE id = N;
 *   exit
 *
 * The database is persisted to the given file.  Restart the REPL and
 * all previously inserted rows are still there.
 *
 * Type Ctrl-D (EOF) or "exit" to quit.
 */

#include "table.h"
#include "sql_parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INPUT_BUF 512

int main(int argc, char *argv[])
{
    const char *db_file = (argc >= 2) ? argv[1] : "test.db";

    printf("Mini-Database REPL\n");
    printf("Database file: %s\n", db_file);
    printf("Type: INSERT INTO rows VALUES (id, 'name', score);\n");
    printf("      SELECT * FROM rows;\n");
    printf("      SELECT * FROM rows WHERE id = N;\n");
    printf("      exit\n\n");

    Table *table = table_open(db_file);
    printf("(%d row(s) loaded from disk)\n\n", table_row_count(table));

    char buf[INPUT_BUF];
    for (;;) {
        printf("db> ");
        fflush(stdout);

        if (!fgets(buf, sizeof(buf), stdin)) {
            /* EOF (Ctrl-D) */
            printf("\n");
            break;
        }

        /* Strip trailing newline */
        size_t len = strlen(buf);
        while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r'))
            buf[--len] = '\0';

        if (len == 0) continue;

        SqlStatement stmt = sql_parse(buf);

        switch (stmt.type) {
        case SQL_EXIT:
            goto done;

        case SQL_INSERT:
            table_insert(table,
                         stmt.insert_id,
                         stmt.insert_name,
                         stmt.insert_score);
            printf("Inserted row (id=%d, name='%s', score=%.2f)\n",
                   stmt.insert_id, stmt.insert_name, stmt.insert_score);
            break;

        case SQL_SELECT_ALL:
            table_select_all(table);
            break;

        case SQL_SELECT_WHERE: {
            Row row;
            if (table_select_by_id(table, stmt.where_id, &row)) {
                printf("  %5s | %-32s | %8s\n", "id", "name", "score");
                printf("  %s\n",
                       "-------+----------------------------------+----------");
                printf("  %5d | %-32s | %8.2f\n",
                       row.id, row.name, row.score);
            } else {
                printf("No row with id=%d\n", stmt.where_id);
            }
            break;
        }

        case SQL_UNKNOWN:
            printf("Error: %s\n", stmt.error);
            break;
        }
    }

done:
    table_close(table);
    printf("Database saved to '%s'. Goodbye.\n", db_file);
    return 0;
}
