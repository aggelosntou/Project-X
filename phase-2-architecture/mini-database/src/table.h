#ifndef TABLE_H
#define TABLE_H

/*
 * table.h — "rows" table built on top of an in-memory B-tree index
 *           and a pager for on-disk persistence.
 *
 * Schema:
 *   id    INT  (primary key, used as B-tree key)
 *   name  CHAR(32)
 *   score DOUBLE
 *   Total row size = 4 + 32 + 8 = 44 bytes, padded to 48 for alignment.
 *
 * Storage model:
 *   - Rows are stored in the pager (page-based file), packed sequentially.
 *   - The B-tree index maps id → row_number (the position in the page file).
 *   - On open, the index is rebuilt by scanning all pages.
 *   - On close, all dirty pages are flushed.
 *
 * Row layout in a page:
 *   Each page holds PAGE_SIZE / ROW_SIZE rows.
 *   page 0, slot 0 → row 0
 *   page 0, slot 1 → row 1
 *   …
 */

#include "btree.h"
#include "pager.h"

#define NAME_LEN   32
#define ROW_SIZE   48    /* id(4) + name(32) + score(8) + pad(4) */
#define ROWS_PER_PAGE  (PAGE_SIZE / ROW_SIZE)   /* 4096/48 = 85 */

typedef struct {
    int    id;
    char   name[NAME_LEN];
    double score;
} Row;

typedef struct {
    BTree *index;      /* id → row_number in pager */
    Pager *pager;
    int    row_count;
    char   db_path[256];
} Table;

/* Open (or create) a table backed by db_file. */
Table *table_open(const char *db_file);

/* Insert a new row.  If id already exists, the call is a no-op and
 * an error is printed. */
void table_insert(Table *t, int id, const char *name, double score);

/* Print all rows ordered by id. */
void table_select_all(Table *t);

/* Find the row with the given id.  Returns 1 and fills *out if found,
 * 0 otherwise. */
int table_select_by_id(Table *t, int id, Row *out);

/* Flush all pages and free all memory. */
void table_close(Table *t);

/* Return number of rows. */
int table_row_count(Table *t);

#endif /* TABLE_H */
