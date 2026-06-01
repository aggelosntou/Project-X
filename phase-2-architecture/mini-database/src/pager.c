/*
 * pager.c — page-based file I/O layer.
 *
 * Page layout on disk:
 *   page 0 starts at byte 0
 *   page 1 starts at byte PAGE_SIZE
 *   page N starts at byte N * PAGE_SIZE
 *
 * In this simplified implementation:
 *   - All pages fit in memory (MAX_PAGES * PAGE_SIZE = 4 MiB).
 *   - Pages are loaded lazily on first access.
 *   - All dirty pages are flushed when pager_close() is called.
 *   - No LRU eviction — if you access more than MAX_PAGES pages the
 *     pager will abort.  A real pager would evict LRU clean pages.
 */

#include "pager.h"
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------
 * Open
 * --------------------------------------------------------------- */
Pager *pager_open(const char *filename)
{
    Pager *p = (Pager *)calloc(1, sizeof(Pager));
    if (!p) { perror("calloc"); exit(EXIT_FAILURE); }

    /* "a+b" creates the file if it doesn't exist; otherwise opens for
     * reading and appending.  We immediately reopen as "r+b" for
     * random read/write access. */
    FILE *f = fopen(filename, "r+b");
    if (!f) {
        /* File doesn't exist yet; create it */
        f = fopen(filename, "w+b");
        if (!f) { perror(filename); exit(EXIT_FAILURE); }
    }
    p->file = f;

    /* Determine how many pages already exist on disk */
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    p->num_pages = (int)(file_size / PAGE_SIZE);

    return p;
}

/* ---------------------------------------------------------------
 * Get page (load from disk if needed)
 * --------------------------------------------------------------- */
uint8_t *pager_get(Pager *p, int page_num)
{
    if (page_num >= MAX_PAGES) {
        fprintf(stderr, "Pager: page %d exceeds MAX_PAGES (%d)\n",
                page_num, MAX_PAGES);
        exit(EXIT_FAILURE);
    }

    if (p->cache[page_num] != NULL)
        return p->cache[page_num];

    /* Allocate cache slot */
    uint8_t *page = (uint8_t *)calloc(1, PAGE_SIZE);
    if (!page) { perror("calloc"); exit(EXIT_FAILURE); }
    p->cache[page_num] = page;

    if (page_num < p->num_pages) {
        /* Load from disk */
        fseek(p->file, (long)page_num * PAGE_SIZE, SEEK_SET);
        size_t nread = fread(page, 1, PAGE_SIZE, p->file);
        (void)nread;  /* partial reads are OK for new pages */
    }
    /* else: new page, already zeroed by calloc */

    if (page_num >= p->num_pages)
        p->num_pages = page_num + 1;

    return page;
}

/* ---------------------------------------------------------------
 * Mark dirty
 * --------------------------------------------------------------- */
void pager_mark_dirty(Pager *p, int page_num)
{
    if (page_num < MAX_PAGES)
        p->dirty[page_num] = 1;
}

/* ---------------------------------------------------------------
 * Flush a single page
 * --------------------------------------------------------------- */
void pager_flush(Pager *p, int page_num)
{
    if (page_num >= MAX_PAGES) return;
    if (!p->cache[page_num]) return;
    if (!p->dirty[page_num]) return;

    fseek(p->file, (long)page_num * PAGE_SIZE, SEEK_SET);
    fwrite(p->cache[page_num], 1, PAGE_SIZE, p->file);
    p->dirty[page_num] = 0;
}

/* ---------------------------------------------------------------
 * Close: flush all dirty pages and release memory
 * --------------------------------------------------------------- */
void pager_close(Pager *p)
{
    if (!p) return;
    for (int i = 0; i < MAX_PAGES; ++i) {
        pager_flush(p, i);
        free(p->cache[i]);
    }
    fclose(p->file);
    free(p);
}

/* ---------------------------------------------------------------
 * Helpers
 * --------------------------------------------------------------- */
int pager_num_pages(Pager *p)
{
    return p->num_pages;
}

int pager_alloc(Pager *p)
{
    int page_num = p->num_pages;
    pager_get(p, page_num);        /* allocates and zeroes */
    pager_mark_dirty(p, page_num);
    return page_num;
}
