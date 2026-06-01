/*
 * string-library/src/mystring.c
 *
 * Implementations of every function declared in mystring.h.
 *
 * Design principles used throughout:
 *   - Work with unsigned char for byte-level operations
 *     (char is implementation-defined as signed or unsigned; casting to
 *      unsigned char gives consistent 0-255 range for comparisons)
 *   - Keep implementations simple and obviously correct rather than optimal
 *   - Add comments that explain WHY each line exists
 */

#include "mystring.h"
#include <stdlib.h>   /* malloc, realloc, free */
#include <assert.h>   /* assert */

/* -----------------------------------------------------------------------
 * my_strlen
 *
 * Walk from the start of s until we hit the null terminator '\0'.
 * Return the number of bytes walked (not including '\0').
 *
 * WHY char *s has type const char *: we promise not to modify the string.
 * ----------------------------------------------------------------------- */
size_t my_strlen(const char *s)
{
    const char *p = s;
    /* Every C string ends with a zero byte.  Keep going until we find it. */
    while (*p != '\0') {
        p++;
    }
    /* p now points to the '\0'.  The distance from s to p is the length. */
    return (size_t)(p - s);
}

/* -----------------------------------------------------------------------
 * my_strcpy
 *
 * Copy each byte of src to dst, INCLUDING the null terminator.
 * This is the one-liner kernel of how strings are "assigned" in C.
 * ----------------------------------------------------------------------- */
char *my_strcpy(char *dst, const char *src)
{
    char *ret = dst;  /* save original dst to return */
    /* The loop copies src[i] to dst[i], then checks if it was '\0'.
     * When '\0' is copied the condition becomes false, and we stop.
     * This pattern is idiomatic C: the assignment IS the loop body. */
    while ((*dst++ = *src++) != '\0') {
        /* intentionally empty — everything happens in the condition */
    }
    return ret;
}

/* -----------------------------------------------------------------------
 * my_strncpy
 *
 * Bounded copy: copy at most n bytes from src to dst.
 * If src runs out before n bytes, pad the remainder of dst with '\0'.
 *
 * WHY the padding?  So that fixed-size fields (e.g., in structs written
 * to disk or sent over the network) don't contain stale garbage bytes.
 *
 * WARNING: if strlen(src) >= n, dst is NOT null-terminated!
 * This is a known pitfall of strncpy — many codebases add an explicit
 * dst[n-1] = '\0' after calling strncpy.
 * ----------------------------------------------------------------------- */
char *my_strncpy(char *dst, const char *src, size_t n)
{
    char *ret = dst;
    while (n > 0) {
        /* Copy one byte from src to dst */
        *dst = *src;
        if (*src == '\0') {
            /* src ended — pad the rest of the destination with '\0' */
            while (--n > 0) {
                *++dst = '\0';
            }
            return ret;
        }
        dst++;
        src++;
        n--;
    }
    return ret;
}

/* -----------------------------------------------------------------------
 * my_strcat
 *
 * Append src to the end of dst.  dst must already be null-terminated
 * and must have enough room for strlen(src) more bytes.
 * ----------------------------------------------------------------------- */
char *my_strcat(char *dst, const char *src)
{
    char *ret = dst;
    /* Step 1: advance dst to its null terminator */
    while (*dst != '\0') {
        dst++;
    }
    /* Step 2: copy src into dst starting at the null terminator position
     * (which overwrites the '\0' and adds a new one at the end) */
    while ((*dst++ = *src++) != '\0') {
        /* empty — same trick as strcpy */
    }
    return ret;
}

/* -----------------------------------------------------------------------
 * my_strcmp
 *
 * Compare strings byte-by-byte.
 * We cast to unsigned char before comparing so that bytes 128-255
 * sort correctly (signed char would treat them as negative numbers).
 *
 * Returns the difference of the first differing bytes, or 0 if equal.
 * ----------------------------------------------------------------------- */
int my_strcmp(const char *a, const char *b)
{
    while (*a != '\0' && *a == *b) {
        a++;
        b++;
    }
    /* Cast to unsigned char — critical for correct comparison of high bytes */
    return (int)(unsigned char)*a - (int)(unsigned char)*b;
}

/* -----------------------------------------------------------------------
 * my_strncmp
 *
 * Like my_strcmp but stops after comparing n bytes.
 * ----------------------------------------------------------------------- */
int my_strncmp(const char *a, const char *b, size_t n)
{
    if (n == 0) return 0;
    while (n > 1 && *a != '\0' && *a == *b) {
        a++;
        b++;
        n--;
    }
    return (int)(unsigned char)*a - (int)(unsigned char)*b;
}

/* -----------------------------------------------------------------------
 * my_strchr
 *
 * Search for the first occurrence of character c in s.
 * The standard says we search including the null terminator — so
 * my_strchr(s, '\0') returns a pointer to the end of the string.
 * ----------------------------------------------------------------------- */
char *my_strchr(const char *s, int c)
{
    unsigned char uc = (unsigned char)c;
    do {
        /* Check BEFORE advancing so we can find '\0' at the end */
        if ((unsigned char)*s == uc) {
            return (char *)s;  /* cast away const — standard behaviour */
        }
    } while (*s++ != '\0');
    return NULL;
}

/* -----------------------------------------------------------------------
 * my_strrchr
 *
 * Search for the LAST occurrence of c.  We can't search backwards
 * without knowing the end, so we scan forward and update our answer
 * every time we find a match.
 * ----------------------------------------------------------------------- */
char *my_strrchr(const char *s, int c)
{
    unsigned char uc = (unsigned char)c;
    const char *last = NULL;
    do {
        if ((unsigned char)*s == uc) {
            last = s;  /* keep updating — the last match wins */
        }
    } while (*s++ != '\0');
    return (char *)last;
}

/* -----------------------------------------------------------------------
 * my_strstr
 *
 * Naive (brute-force) substring search: O(n*m) time.
 * Real implementations use Boyer-Moore or KMP for O(n+m), but the naive
 * version is much easier to understand.
 * ----------------------------------------------------------------------- */
char *my_strstr(const char *haystack, const char *needle)
{
    /* Empty needle always matches at the start — this is standard behaviour */
    if (*needle == '\0') {
        return (char *)haystack;
    }
    while (*haystack != '\0') {
        /* Try to match needle starting at this position in haystack */
        const char *h = haystack;
        const char *n = needle;
        while (*n != '\0' && *h == *n) {
            h++;
            n++;
        }
        if (*n == '\0') {
            /* Reached the end of needle — full match found */
            return (char *)haystack;
        }
        haystack++;
    }
    return NULL;
}

/* -----------------------------------------------------------------------
 * my_memcpy
 *
 * Copy n bytes from src to dst.  Regions must NOT overlap.
 * We use unsigned char * because:
 *   1. It is the correct type for raw byte access in C
 *   2. It avoids strict aliasing issues (unsigned char is special in the
 *      standard: it is allowed to alias any other type)
 * ----------------------------------------------------------------------- */
void *my_memcpy(void *dst, const void *src, size_t n)
{
    unsigned char       *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    while (n--) {
        *d++ = *s++;
    }
    return dst;
}

/* -----------------------------------------------------------------------
 * my_memmove
 *
 * Like my_memcpy but safe when src and dst overlap.
 *
 * WHY overlap is a problem for memcpy:
 *   If dst < src and they overlap, copying forward is fine.
 *   If dst > src and they overlap:
 *     src = [A B C D E]
 *     dst starts 2 bytes into src → [_ _ A B C D E]
 *     Copying forward: write A at dst[0], write B at dst[1],
 *     then read src[2] — but we just overwrote it!
 *
 * Solution: if dst > src (and they might overlap), copy BACKWARD.
 * ----------------------------------------------------------------------- */
void *my_memmove(void *dst, const void *src, size_t n)
{
    unsigned char       *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;

    if (d == s || n == 0) {
        return dst;
    }

    if (d < s || d >= s + n) {
        /* dst is before src, or completely after src+n — no overlap.
         * Forward copy is safe. */
        while (n--) {
            *d++ = *s++;
        }
    } else {
        /* dst overlaps with src from the right side.
         * Copy BACKWARD: start from the end to avoid clobbering src. */
        d += n;
        s += n;
        while (n--) {
            *--d = *--s;
        }
    }
    return dst;
}

/* -----------------------------------------------------------------------
 * my_memset
 *
 * Fill n bytes starting at s with the byte value c.
 * The int c is truncated to unsigned char, which is why:
 *   my_memset(buf, 0xFF, n) and my_memset(buf, -1, n) do the same thing.
 * ----------------------------------------------------------------------- */
void *my_memset(void *s, int c, size_t n)
{
    unsigned char *p = (unsigned char *)s;
    unsigned char  uc = (unsigned char)c;
    while (n--) {
        *p++ = uc;
    }
    return s;
}

/* -----------------------------------------------------------------------
 * my_memcmp
 *
 * Compare n bytes of a and b, treating them as unsigned char arrays.
 * Unsigned comparison is required by the standard — otherwise bytes
 * > 127 would compare as negative.
 * ----------------------------------------------------------------------- */
int my_memcmp(const void *a, const void *b, size_t n)
{
    const unsigned char *pa = (const unsigned char *)a;
    const unsigned char *pb = (const unsigned char *)b;
    while (n--) {
        if (*pa != *pb) {
            return (int)*pa - (int)*pb;
        }
        pa++;
        pb++;
    }
    return 0;
}

/* -----------------------------------------------------------------------
 * DynStr — dynamic string (stretchy buffer)
 *
 * Growth strategy: when we need more space, double the capacity.
 * This ensures that N appends cost O(N) total — amortized O(1) each.
 *
 * WHY doubling? If we grew by a fixed amount (say, +64 bytes each time),
 * appending N bytes to an empty string would require N/64 reallocations,
 * each copying all existing data → O(N^2) total copies.
 * Doubling gives O(N) total copies (geometric series: 1+2+4+...+N < 2N).
 * ----------------------------------------------------------------------- */
#define DYNSTR_INITIAL_CAP 16  /* start small; will grow as needed */

DynStr dynstr_new(void)
{
    DynStr ds;
    ds.data = (char *)malloc(DYNSTR_INITIAL_CAP);
    if (ds.data == NULL) {
        /* Allocation failed — return a zeroed struct.
         * Callers should check ds.data != NULL before using. */
        ds.len = 0;
        ds.cap = 0;
        return ds;
    }
    ds.data[0] = '\0';  /* always null-terminated */
    ds.len = 0;
    ds.cap = DYNSTR_INITIAL_CAP;
    return ds;
}

/* Internal helper: ensure ds has room for at least 'needed' bytes
 * (including the null terminator). */
static void dynstr_ensure_cap(DynStr *ds, size_t needed)
{
    if (needed <= ds->cap) return;

    /* Double until we have enough room */
    size_t new_cap = ds->cap == 0 ? DYNSTR_INITIAL_CAP : ds->cap;
    while (new_cap < needed) {
        new_cap *= 2;
    }

    char *new_data = (char *)realloc(ds->data, new_cap);
    if (new_data == NULL) {
        /* realloc failed — in production code you'd propagate the error.
         * Here we abort to keep the example simple. */
        assert(0 && "dynstr: realloc failed");
    }
    ds->data = new_data;
    ds->cap  = new_cap;
}

void dynstr_append_char(DynStr *ds, char c)
{
    /* We need len + 1 chars plus the null terminator = len + 2 bytes */
    dynstr_ensure_cap(ds, ds->len + 2);
    ds->data[ds->len]     = c;
    ds->data[ds->len + 1] = '\0';
    ds->len++;
}

void dynstr_append_str(DynStr *ds, const char *s)
{
    size_t slen = my_strlen(s);
    /* Need room for existing content + new string + null terminator */
    dynstr_ensure_cap(ds, ds->len + slen + 1);
    my_memcpy(ds->data + ds->len, s, slen + 1);  /* +1 copies the '\0' */
    ds->len += slen;
}

void dynstr_free(DynStr *ds)
{
    free(ds->data);
    ds->data = NULL;
    ds->len  = 0;
    ds->cap  = 0;
}
