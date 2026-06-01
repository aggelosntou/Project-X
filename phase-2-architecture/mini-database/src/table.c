/*
 * table.c — Row storage and retrieval.
 *
 * Row serialisation format (48 bytes per row):
 *   Offset  0: int32_t id          (4 bytes)
 *   Offset  4: char    name[32]    (32 bytes, NUL-padded)
 *   Offset 36: double  score       (8 bytes)
 *   Offset 44: padding             (4 bytes, zeroed)
 *
 * Row N lives at:
 *   page   = N / ROWS_PER_PAGE
 *   offset = (N % ROWS_PER_PAGE) * ROW_SIZE
 */

#include "table.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------
 * Serialise / deserialise
 * --------------------------------------------------------------- */
static void row_serialize(const Row *row, uint8_t *dest)
{
    memset(dest, 0, ROW_SIZE);
    memcpy(dest,      &row->id,    sizeof(int));
    memcpy(dest + 4,  row->name,   NAME_LEN);
    memcpy(dest + 36, &row->score, sizeof(double));
}

static void row_deserialize(const uint8_t *src, Row *row)
{
    memcpy(&row->id,    src,      sizeof(int));
    memcpy(row->name,   src + 4,  NAME_LEN);
    row->name[NAME_LEN - 1] = '\0';
    memcpy(&row->score, src + 36, sizeof(double));
}

/* ---------------------------------------------------------------
 * Get pointer to row slot N in the pager
 * --------------------------------------------------------------- */
static uint8_t *row_slot(Table *t, int row_num)
{
    int page_num = row_num / ROWS_PER_PAGE;
    int offset   = (row_num % ROWS_PER_PAGE) * ROW_SIZE;
    uint8_t *page = pager_get(t->pager, page_num);
    return page + offset;
}

/* ---------------------------------------------------------------
 * Rebuild index from all pages on disk
 * --------------------------------------------------------------- */
static void rebuild_index(Table *t)
{
    int total_pages = pager_num_pages(t->pager);
    t->row_count = 0;

    for (int pg = 0; pg < total_pages; ++pg) {
        uint8_t *page = pager_get(t->pager, pg);
        for (int slot = 0; slot < ROWS_PER_PAGE; ++slot) {
            uint8_t *slot_ptr = page + slot * ROW_SIZE;

            /* A row with id == 0 and empty name is treated as empty slot.
             * We use id > 0 as the "occupied" marker. */
            int id;
            memcpy(&id, slot_ptr, sizeof(int));
            if (id <= 0) continue;

            int row_num = pg * ROWS_PER_PAGE + slot;
            btree_insert(t->index, id, (long)row_num);
            t->row_count++;
        }
    }
}

/* ---------------------------------------------------------------
 * Table open
 * --------------------------------------------------------------- */
Table *table_open(const char *db_file)
{
    Table *t = (Table *)calloc(1, sizeof(Table));
    if (!t) { perror("calloc"); exit(EXIT_FAILURE); }

    strncpy(t->db_path, db_file, sizeof(t->db_path) - 1);
    t->pager = pager_open(db_file);
    t->index = btree_new();

    rebuild_index(t);
    return t;
}

/* ---------------------------------------------------------------
 * Insert
 * --------------------------------------------------------------- */
void table_insert(Table *t, int id, const char *name, double score)
{
    if (id <= 0) {
        fprintf(stderr, "INSERT error: id must be > 0 (got %d)\n", id);
        return;
    }

    long existing;
    if (btree_search(t->index, id, &existing)) {
        fprintf(stderr, "INSERT error: id %d already exists\n", id);
        return;
    }

    int row_num = t->row_count;
    int page_num = row_num / ROWS_PER_PAGE;

    uint8_t *slot = row_slot(t, row_num);
    Row row;
    row.id = id;
    strncpy(row.name, name, NAME_LEN - 1);
    row.name[NAME_LEN - 1] = '\0';
    row.score = score;
    row_serialize(&row, slot);

    pager_mark_dirty(t->pager, page_num);
    btree_insert(t->index, id, (long)row_num);
    t->row_count++;
}

/* ---------------------------------------------------------------
 * Select all (in id order via in-order B-tree traversal)
 * --------------------------------------------------------------- */

/* We need an in-order B-tree walk.  Provide a simple recursive one. */
static void inorder_print(BTreeNode *n, Table *t)
{
    if (!n) return;
    for (int i = 0; i <= n->num_keys; ++i) {
        if (!n->is_leaf)
            inorder_print(n->children[i], t);
        if (i < n->num_keys) {
            int row_num = (int)n->values[i];
            uint8_t *slot = row_slot(t, row_num);
            Row row;
            row_deserialize(slot, &row);
            printf("  %5d | %-32s | %8.2f\n",
                   row.id, row.name, row.score);
        }
    }
}

void table_select_all(Table *t)
{
    printf("  %5s | %-32s | %8s\n", "id", "name", "score");
    printf("  %s\n", "-------+----------------------------------+----------");
    inorder_print(t->index->root, t);
    printf("  %d row(s)\n", t->row_count);
}

/* ---------------------------------------------------------------
 * Select by id
 * --------------------------------------------------------------- */
int table_select_by_id(Table *t, int id, Row *out)
{
    long row_num;
    if (!btree_search(t->index, id, &row_num)) return 0;

    uint8_t *slot = row_slot(t, (int)row_num);
    row_deserialize(slot, out);
    return 1;
}

/* ---------------------------------------------------------------
 * Close
 * --------------------------------------------------------------- */
void table_close(Table *t)
{
    if (!t) return;
    pager_close(t->pager);
    btree_free(t->index);
    free(t);
}

int table_row_count(Table *t)
{
    return t->row_count;
}
