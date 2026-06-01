/*
 * string-library/src/test_mystring.c
 *
 * Comprehensive test suite for mystring.h / mystring.c.
 *
 * Uses assert() for simple pass/fail checks.  If any assertion fails,
 * the program prints the location and aborts.  If it exits normally
 * (exit code 0), all tests passed.
 *
 * Run with AddressSanitizer to catch buffer overruns:
 *   gcc -fsanitize=address -o test_mystring test_mystring.c mystring.c
 *   ./test_mystring
 */

#include "mystring.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>   /* for reference comparisons */

/* -----------------------------------------------------------------------
 * Helper macro: run a test and print a green check or fail message
 * ----------------------------------------------------------------------- */
#define TEST(name) \
    do { printf("  %-40s", name); } while(0)
#define PASS() \
    do { printf("PASS\n"); } while(0)

/* -----------------------------------------------------------------------
 * my_strlen tests
 * ----------------------------------------------------------------------- */
static void test_strlen(void)
{
    printf("\n=== my_strlen ===\n");

    TEST("empty string");
    assert(my_strlen("") == 0);
    PASS();

    TEST("single char");
    assert(my_strlen("a") == 1);
    PASS();

    TEST("hello world");
    assert(my_strlen("hello world") == 11);
    PASS();

    /* A string with embedded nulls — strlen stops at the FIRST one */
    TEST("stops at first null");
    const char embedded[] = {'a', 'b', '\0', 'c', 'd'};
    assert(my_strlen(embedded) == 2);
    PASS();

    TEST("matches stdlib strlen");
    const char *s = "The quick brown fox";
    assert(my_strlen(s) == strlen(s));
    PASS();
}

/* -----------------------------------------------------------------------
 * my_strcpy tests
 * ----------------------------------------------------------------------- */
static void test_strcpy(void)
{
    printf("\n=== my_strcpy ===\n");
    char buf[128];

    TEST("basic copy");
    my_strcpy(buf, "hello");
    assert(my_strcmp(buf, "hello") == 0);
    PASS();

    TEST("copies null terminator");
    my_strcpy(buf, "abc");
    assert(buf[3] == '\0');
    PASS();

    TEST("empty string");
    my_strcpy(buf, "");
    assert(buf[0] == '\0');
    PASS();

    TEST("returns dst");
    char *ret = my_strcpy(buf, "test");
    assert(ret == buf);
    PASS();

    TEST("overwrites longer string");
    my_strcpy(buf, "longstring");
    my_strcpy(buf, "hi");
    assert(buf[2] == '\0');
    assert(my_strlen(buf) == 2);
    PASS();
}

/* -----------------------------------------------------------------------
 * my_strncpy tests
 * ----------------------------------------------------------------------- */
static void test_strncpy(void)
{
    printf("\n=== my_strncpy ===\n");
    char buf[16];

    TEST("copy n < len: no null term added");
    my_memset(buf, 0xFF, sizeof(buf));
    my_strncpy(buf, "hello", 3);
    assert(buf[0] == 'h' && buf[1] == 'e' && buf[2] == 'l');
    /* buf[3] should still be 0xFF — strncpy doesn't add null if n < len */
    assert((unsigned char)buf[3] == 0xFF);
    PASS();

    TEST("copy n == len: pads with null");
    my_memset(buf, 0xFF, sizeof(buf));
    my_strncpy(buf, "hi", 8);
    assert(buf[0] == 'h' && buf[1] == 'i');
    /* Remaining bytes should be padded with '\0' */
    for (int i = 2; i < 8; i++) {
        assert(buf[i] == '\0');
    }
    PASS();

    TEST("n = 0: copies nothing");
    my_memset(buf, 'X', sizeof(buf));
    my_strncpy(buf, "hello", 0);
    assert(buf[0] == 'X');
    PASS();
}

/* -----------------------------------------------------------------------
 * my_strcat tests
 * ----------------------------------------------------------------------- */
static void test_strcat(void)
{
    printf("\n=== my_strcat ===\n");
    char buf[64];

    TEST("basic append");
    my_strcpy(buf, "hello");
    my_strcat(buf, " world");
    assert(my_strcmp(buf, "hello world") == 0);
    PASS();

    TEST("append to empty");
    my_strcpy(buf, "");
    my_strcat(buf, "foo");
    assert(my_strcmp(buf, "foo") == 0);
    PASS();

    TEST("append empty string");
    my_strcpy(buf, "bar");
    my_strcat(buf, "");
    assert(my_strcmp(buf, "bar") == 0);
    PASS();

    TEST("chained appends");
    my_strcpy(buf, "a");
    my_strcat(buf, "b");
    my_strcat(buf, "c");
    assert(my_strcmp(buf, "abc") == 0);
    PASS();
}

/* -----------------------------------------------------------------------
 * my_strcmp tests
 * ----------------------------------------------------------------------- */
static void test_strcmp(void)
{
    printf("\n=== my_strcmp ===\n");

    TEST("equal strings → 0");
    assert(my_strcmp("abc", "abc") == 0);
    PASS();

    TEST("less than → negative");
    assert(my_strcmp("abc", "abd") < 0);
    PASS();

    TEST("greater than → positive");
    assert(my_strcmp("abd", "abc") > 0);
    PASS();

    TEST("empty strings equal");
    assert(my_strcmp("", "") == 0);
    PASS();

    TEST("prefix less than full");
    assert(my_strcmp("abc", "abcd") < 0);
    PASS();

    TEST("full greater than prefix");
    assert(my_strcmp("abcd", "abc") > 0);
    PASS();

    TEST("matches stdlib strcmp sign");
    assert((my_strcmp("hello", "world") < 0) == (strcmp("hello", "world") < 0));
    PASS();
}

/* -----------------------------------------------------------------------
 * my_strncmp tests
 * ----------------------------------------------------------------------- */
static void test_strncmp(void)
{
    printf("\n=== my_strncmp ===\n");

    TEST("equal for n bytes");
    assert(my_strncmp("abcXXX", "abcYYY", 3) == 0);
    PASS();

    TEST("differ within n bytes");
    assert(my_strncmp("abcX", "abcY", 4) != 0);
    PASS();

    TEST("n = 0 always equal");
    assert(my_strncmp("totally", "different", 0) == 0);
    PASS();
}

/* -----------------------------------------------------------------------
 * my_strchr / my_strrchr tests
 * ----------------------------------------------------------------------- */
static void test_strchr(void)
{
    printf("\n=== my_strchr / my_strrchr ===\n");

    const char *s = "hello world";

    TEST("strchr: find 'o' first occurrence");
    char *p = my_strchr(s, 'o');
    assert(p != NULL && *p == 'o' && p == s + 4);
    PASS();

    TEST("strchr: char not found → NULL");
    assert(my_strchr(s, 'z') == NULL);
    PASS();

    TEST("strchr: find null terminator");
    p = my_strchr(s, '\0');
    assert(p != NULL && p == s + my_strlen(s));
    PASS();

    TEST("strrchr: find 'o' last occurrence");
    p = my_strrchr(s, 'o');
    assert(p != NULL && *p == 'o' && p == s + 7);
    PASS();

    TEST("strrchr: char not found → NULL");
    assert(my_strrchr(s, 'z') == NULL);
    PASS();

    TEST("strrchr: single character");
    p = my_strrchr("a", 'a');
    assert(p != NULL);
    PASS();
}

/* -----------------------------------------------------------------------
 * my_strstr tests
 * ----------------------------------------------------------------------- */
static void test_strstr(void)
{
    printf("\n=== my_strstr ===\n");

    TEST("find substring at start");
    const char *h = "hello world";
    char *p = my_strstr(h, "hello");
    assert(p == h);
    PASS();

    TEST("find substring in middle");
    p = my_strstr(h, "world");
    assert(p == h + 6);
    PASS();

    TEST("substring not found → NULL");
    assert(my_strstr(h, "xyz") == NULL);
    PASS();

    TEST("empty needle → haystack");
    p = my_strstr(h, "");
    assert(p == h);
    PASS();

    TEST("needle == haystack");
    assert(my_strstr("abc", "abc") != NULL);
    PASS();

    TEST("needle longer than haystack → NULL");
    assert(my_strstr("hi", "hello") == NULL);
    PASS();
}

/* -----------------------------------------------------------------------
 * my_memcpy tests
 * ----------------------------------------------------------------------- */
static void test_memcpy(void)
{
    printf("\n=== my_memcpy ===\n");
    unsigned char src[8] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    unsigned char dst[8] = {0};

    TEST("copy 8 bytes");
    my_memcpy(dst, src, 8);
    assert(my_memcmp(dst, src, 8) == 0);
    PASS();

    TEST("copy 0 bytes: dst unchanged");
    my_memset(dst, 0xAB, 8);
    my_memcpy(dst, src, 0);
    assert(dst[0] == 0xAB);
    PASS();

    TEST("copy partial");
    my_memset(dst, 0, 8);
    my_memcpy(dst, src, 3);
    assert(dst[0] == 0x01 && dst[1] == 0x02 && dst[2] == 0x03);
    assert(dst[3] == 0);
    PASS();

    TEST("returns dst");
    void *ret = my_memcpy(dst, src, 8);
    assert(ret == dst);
    PASS();
}

/* -----------------------------------------------------------------------
 * my_memmove tests — overlap is the key case
 * ----------------------------------------------------------------------- */
static void test_memmove(void)
{
    printf("\n=== my_memmove ===\n");
    unsigned char buf[16];

    TEST("non-overlapping: same as memcpy");
    my_memset(buf, 0, sizeof(buf));
    buf[0] = 'A'; buf[1] = 'B'; buf[2] = 'C';
    my_memmove(buf + 8, buf, 3);
    assert(buf[8] == 'A' && buf[9] == 'B' && buf[10] == 'C');
    PASS();

    /* Forward overlap: src=[0..4], dst=[2..6] — dst > src so must copy backward
     * buf = [1 2 3 4 5 _ _ _ ...] → move [0..4] to [2..6]
     * Result: [1 2 1 2 3 4 5 _ ...] */
    TEST("forward overlap (dst > src)");
    for (int i = 0; i < 5; i++) buf[i] = (unsigned char)(i + 1);
    my_memmove(buf + 2, buf, 5);
    assert(buf[2] == 1 && buf[3] == 2 && buf[4] == 3 &&
           buf[5] == 4 && buf[6] == 5);
    PASS();

    /* Backward overlap: src=[2..6], dst=[0..4] — dst < src so forward copy ok
     * buf = [_ _ 1 2 3 4 5 _ ...] → move [2..6] to [0..4] */
    TEST("backward overlap (dst < src)");
    my_memset(buf, 0, sizeof(buf));
    for (int i = 0; i < 5; i++) buf[i + 2] = (unsigned char)(i + 1);
    my_memmove(buf, buf + 2, 5);
    assert(buf[0] == 1 && buf[1] == 2 && buf[2] == 3 &&
           buf[3] == 4 && buf[4] == 5);
    PASS();

    TEST("move to same location: no change");
    buf[0] = 42;
    my_memmove(buf, buf, 4);
    assert(buf[0] == 42);
    PASS();
}

/* -----------------------------------------------------------------------
 * my_memset tests
 * ----------------------------------------------------------------------- */
static void test_memset(void)
{
    printf("\n=== my_memset ===\n");
    char buf[16];

    TEST("fill with zero");
    my_memset(buf, 0, sizeof(buf));
    for (int i = 0; i < 16; i++) assert(buf[i] == 0);
    PASS();

    TEST("fill with 0xFF");
    my_memset(buf, 0xFF, sizeof(buf));
    for (int i = 0; i < 16; i++) assert((unsigned char)buf[i] == 0xFF);
    PASS();

    TEST("fill partial");
    my_memset(buf, 0, 16);
    my_memset(buf + 4, 'X', 4);
    assert(buf[3] == 0 && buf[4] == 'X' && buf[7] == 'X' && buf[8] == 0);
    PASS();

    TEST("n = 0: no change");
    buf[0] = 'Z';
    my_memset(buf, 0, 0);
    assert(buf[0] == 'Z');
    PASS();
}

/* -----------------------------------------------------------------------
 * my_memcmp tests
 * ----------------------------------------------------------------------- */
static void test_memcmp(void)
{
    printf("\n=== my_memcmp ===\n");

    TEST("equal regions → 0");
    assert(my_memcmp("abcde", "abcde", 5) == 0);
    PASS();

    TEST("first < second → negative");
    assert(my_memcmp("abc", "abd", 3) < 0);
    PASS();

    TEST("first > second → positive");
    assert(my_memcmp("abd", "abc", 3) > 0);
    PASS();

    TEST("n = 0 always equal");
    assert(my_memcmp("different", "completely", 0) == 0);
    PASS();

    TEST("high bytes compared as unsigned");
    unsigned char a[2] = {0x80, 0x00};
    unsigned char b[2] = {0x01, 0x00};
    assert(my_memcmp(a, b, 2) > 0);  /* 0x80 > 0x01 as unsigned */
    PASS();
}

/* -----------------------------------------------------------------------
 * DynStr tests
 * ----------------------------------------------------------------------- */
static void test_dynstr(void)
{
    printf("\n=== DynStr (dynamic string / stretchy buffer) ===\n");

    TEST("new: empty, len=0");
    DynStr ds = dynstr_new();
    assert(ds.data != NULL);
    assert(ds.len == 0);
    assert(ds.data[0] == '\0');
    PASS();

    TEST("append_char: single char");
    dynstr_append_char(&ds, 'H');
    assert(ds.len == 1 && ds.data[0] == 'H' && ds.data[1] == '\0');
    PASS();

    TEST("append_str: string");
    dynstr_append_str(&ds, "ello");
    assert(ds.len == 5 && my_strcmp(ds.data, "Hello") == 0);
    PASS();

    TEST("append_str: force realloc");
    /* Build a long string to trigger buffer growth */
    DynStr big = dynstr_new();
    for (int i = 0; i < 200; i++) {
        dynstr_append_char(&big, (char)('a' + i % 26));
    }
    assert(big.len == 200);
    assert(big.cap >= 201);
    /* Verify null termination */
    assert(big.data[200] == '\0');
    dynstr_free(&big);
    PASS();

    TEST("append_str: empty string");
    size_t old_len = ds.len;
    dynstr_append_str(&ds, "");
    assert(ds.len == old_len);
    PASS();

    TEST("free: sets data to NULL");
    dynstr_free(&ds);
    assert(ds.data == NULL && ds.len == 0 && ds.cap == 0);
    PASS();

    TEST("building a sentence char by char");
    DynStr sentence = dynstr_new();
    const char *words[] = {"The", " ", "quick", " ", "fox"};
    for (int i = 0; i < 5; i++) {
        dynstr_append_str(&sentence, words[i]);
    }
    assert(my_strcmp(sentence.data, "The quick fox") == 0);
    dynstr_free(&sentence);
    PASS();
}

/* -----------------------------------------------------------------------
 * main — run all tests
 * ----------------------------------------------------------------------- */
int main(void)
{
    printf("Running mystring test suite...\n");

    test_strlen();
    test_strcpy();
    test_strncpy();
    test_strcat();
    test_strcmp();
    test_strncmp();
    test_strchr();
    test_strstr();
    test_memcpy();
    test_memmove();
    test_memset();
    test_memcmp();
    test_dynstr();

    printf("\n===========================================\n");
    printf("All tests PASSED.\n");
    printf("===========================================\n");
    return 0;
}
