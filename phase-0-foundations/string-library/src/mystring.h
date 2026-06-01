/*
 * string-library/src/mystring.h
 *
 * Public interface for our reimplementation of <string.h> and a
 * simple dynamic string (stretchy buffer).
 *
 * WHY reimplementing these is educational:
 *   - Forces you to think about null terminators, byte-by-byte access,
 *     and the difference between safe/unsafe operations.
 *   - The implementations are 10-30 lines each — demystifying functions
 *     that feel like magic when you first use them.
 */

#ifndef MYSTRING_H
#define MYSTRING_H

#include <stddef.h>  /* for size_t */

/* -----------------------------------------------------------------------
 * Standard string functions (analogous to <string.h>)
 * ----------------------------------------------------------------------- */

/* Returns the number of bytes in s before the null terminator '\0'.
 * Does NOT include the terminator in the count. */
size_t my_strlen(const char *s);

/* Copies src (including '\0') to dst.  Returns dst.
 * UNSAFE: dst must be large enough.  Use my_strncpy for safety. */
char *my_strcpy(char *dst, const char *src);

/* Copies at most n bytes from src to dst.
 * If src is shorter than n bytes, pads the rest of dst with '\0'.
 * WARNING: if src is >= n bytes, dst may NOT be null-terminated! */
char *my_strncpy(char *dst, const char *src, size_t n);

/* Appends src to the end of dst.  dst must have room.  Returns dst. */
char *my_strcat(char *dst, const char *src);

/* Lexicographic comparison.
 * Returns: < 0 if a < b, 0 if a == b, > 0 if a > b. */
int   my_strcmp(const char *a, const char *b);

/* Bounded lexicographic comparison — compares at most n bytes. */
int   my_strncmp(const char *a, const char *b, size_t n);

/* Finds the first occurrence of character c in string s.
 * Returns pointer to the character, or NULL if not found.
 * Note: c is int because the standard passes characters as int. */
char *my_strchr(const char *s, int c);

/* Finds the last occurrence of character c in string s.
 * Returns pointer, or NULL if not found. */
char *my_strrchr(const char *s, int c);

/* Finds first occurrence of needle within haystack.
 * Returns pointer to start of match, or NULL. */
char *my_strstr(const char *haystack, const char *needle);

/* -----------------------------------------------------------------------
 * Memory functions (analogous to <string.h> mem* family)
 * ----------------------------------------------------------------------- */

/* Copies n bytes from src to dst.  Regions must NOT overlap.
 * Returns dst. */
void *my_memcpy(void *dst, const void *src, size_t n);

/* Copies n bytes from src to dst, handling overlap correctly.
 * When regions overlap, copies backward to avoid clobbering src.
 * Returns dst. */
void *my_memmove(void *dst, const void *src, size_t n);

/* Sets n bytes starting at s to (unsigned char)c.  Returns s. */
void *my_memset(void *s, int c, size_t n);

/* Compares n bytes of a and b lexicographically (as unsigned char).
 * Returns: < 0, 0, or > 0. */
int   my_memcmp(const void *a, const void *b, size_t n);

/* -----------------------------------------------------------------------
 * Dynamic string (stretchy buffer)
 *
 * A DynStr owns a heap-allocated buffer that grows as needed.
 * The pattern is similar to C++'s std::string or Python's str.
 *
 * Invariant: data[len] == '\0' always (when len < cap).
 * ----------------------------------------------------------------------- */

typedef struct {
    char   *data;  /* heap-allocated character buffer                  */
    size_t  len;   /* number of characters (excluding null terminator) */
    size_t  cap;   /* total allocated bytes (including the '\0' slot)  */
} DynStr;

/* Creates a new empty DynStr.  Returns it by value for simplicity.
 * Call dynstr_free() when done. */
DynStr  dynstr_new(void);

/* Appends a single character to ds.  Resizes if necessary. */
void    dynstr_append_char(DynStr *ds, char c);

/* Appends a null-terminated string to ds.  Resizes if necessary. */
void    dynstr_append_str(DynStr *ds, const char *s);

/* Frees the internal buffer and resets the struct to empty state. */
void    dynstr_free(DynStr *ds);

#endif /* MYSTRING_H */
