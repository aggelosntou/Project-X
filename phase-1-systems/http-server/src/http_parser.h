/*
 * http_parser.h — Structures and functions for parsing HTTP/1.1 requests
 * and building HTTP responses.
 *
 * HTTP/1.1 request format:
 *
 *   METHOD PATH HTTP/version\r\n
 *   Header-Name: Header-Value\r\n
 *   ...
 *   \r\n
 *   [optional body]
 *
 * Example:
 *   GET /hello HTTP/1.1\r\n
 *   Host: localhost:8080\r\n
 *   Accept: text/html\r\n
 *   \r\n
 *
 * HTTP/1.1 response format:
 *
 *   HTTP/1.1 STATUS_CODE STATUS_TEXT\r\n
 *   Header-Name: Header-Value\r\n
 *   \r\n
 *   [body]
 */

#ifndef HTTP_PARSER_H
#define HTTP_PARSER_H

#include <stddef.h>

/* ── Limits ──────────────────────────────────────────────────────────────── */

#define HTTP_MAX_METHOD    16
#define HTTP_MAX_PATH     256
#define HTTP_MAX_VERSION   16
#define HTTP_MAX_HEADERS   32
#define HTTP_MAX_HDR_NAME 128
#define HTTP_MAX_HDR_VAL  512
#define HTTP_MAX_BODY    8192

/* ── HttpRequest struct ──────────────────────────────────────────────────── */

typedef struct {
    char method[HTTP_MAX_METHOD];
    char path[HTTP_MAX_PATH];
    char http_version[HTTP_MAX_VERSION];

    /* Headers stored as [name][value] pairs */
    char headers[HTTP_MAX_HEADERS][2][HTTP_MAX_HDR_VAL];
    int  header_count;

    char body[HTTP_MAX_BODY];
    int  body_len;
} HttpRequest;

/* ── Public API ─────────────────────────────────────────────────────────── */

/**
 * parse_request — parse a raw HTTP request from buf[0..len-1].
 *
 * Fills in *req.  Returns 0 on success, -1 on parse error.
 *
 * The buffer does not need to be NUL-terminated but must be at least `len`
 * bytes readable.
 */
int parse_request(const char *buf, int len, HttpRequest *req);

/**
 * build_response — write an HTTP/1.1 response into out_buf.
 *
 * status_code  e.g. 200
 * status_text  e.g. "OK"
 * content_type e.g. "text/html; charset=utf-8"
 * body         response body (may be NULL for status-only responses)
 * body_len     length of body in bytes
 * out_buf      caller-provided buffer
 * out_size     size of out_buf
 *
 * Returns the total number of bytes written to out_buf, or -1 on error.
 */
int build_response(int status_code, const char *status_text,
                   const char *content_type,
                   const char *body, int body_len,
                   char *out_buf, int out_size);

/**
 * http_find_header — look up a header value by name (case-insensitive).
 *
 * Returns a pointer to the value string, or NULL if not found.
 */
const char *http_find_header(const HttpRequest *req, const char *name);

#endif /* HTTP_PARSER_H */
