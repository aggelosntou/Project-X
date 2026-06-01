#ifndef PAGER_H
#define PAGER_H

/*
 * pager.h — page-based file I/O layer.
 *
 * A "page" is the unit of I/O between disk and memory.  The OS and disk
 * controller work in page-sized chunks (typically 4096 bytes), so aligning
 * database operations to page boundaries means every read/write is one
 * single OS I/O call.
 *
 * The pager maintains a simple write-through cache: pages are loaded from
 * disk on first access and written back when marked dirty + flushed.
 *
 * In a production database (SQLite, Postgres) the pager is the layer that
 * implements:
 *   - Buffer pool management (LRU eviction when cache is full)
 *   - Write-ahead logging (WAL) for crash recovery
 *   - Fsync calls to guarantee durability
 *
 * This implementation is intentionally simple: it holds up to MAX_PAGES
 * pages in memory and writes them back on pager_close().
 */

#include <stdint.h>
#include <stdio.h>

#define PAGE_SIZE  4096
#define MAX_PAGES  1024

typedef struct {
    FILE    *file;
    uint8_t *cache[MAX_PAGES];  /* NULL if page not in cache */
    int      dirty[MAX_PAGES];  /* 1 if page has been modified */
    int      num_pages;         /* highest page number allocated + 1 */
} Pager;

/* Open (or create) a database file and return a Pager. */
Pager   *pager_open(const char *filename);

/* Return a pointer to the in-memory copy of page_num.
 * Loads from disk if not already cached.
 * Allocates a new page (zeroed) if page_num >= num_pages. */
uint8_t *pager_get(Pager *p, int page_num);

/* Mark page_num as needing to be written back to disk. */
void     pager_mark_dirty(Pager *p, int page_num);

/* Write a single dirty page to disk immediately. */
void     pager_flush(Pager *p, int page_num);

/* Flush all dirty pages and close the file. */
void     pager_close(Pager *p);

/* Return the number of pages currently allocated. */
int      pager_num_pages(Pager *p);

/* Allocate and return the index of a new blank page. */
int      pager_alloc(Pager *p);

#endif /* PAGER_H */
