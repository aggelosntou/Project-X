/*
 * http_parser.c — HTTP/1.1 request parser and response builder.
 *
 * HTTP parsing in real-world servers (nginx, Apache) is often the hottest
 * path and heavily optimised with SIMD.  Our parser is a simple state-machine
 * that is correct and readable, not fast.
 */

#define _POSIX_C_SOURCE 200809L

#include "http_parser.h"

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <stdarg.h>

/* ── Internal helpers ────────────────────────────────────────────────────── */

/*
 * find_crlf — find the position of "\r\n" in buf[0..len-1].
 * Returns index of '\r', or -1 if not found.
 */
static int find_crlf(const char *buf, int len) {
    for (int i = 0; i < len - 1; i++) {
        if (buf[i] == '\r' && buf[i+1] == '\n') return i;
    }
    return -1;
}

/*
 * str_tolower_cmp — case-insensitive strcmp.
 */
static int str_ci_eq(const char *a, const char *b) {
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
        a++; b++;
    }
    return *a == '\0' && *b == '\0';
}

/* ── parse_request ───────────────────────────────────────────────────────── */

int parse_request(const char *buf, int len, HttpRequest *req) {
    memset(req, 0, sizeof(*req));
    if (len <= 0) return -1;

    int pos = 0;

    /* ── Request line: "METHOD PATH VERSION\r\n" ─────────────────────── */
    int crlf = find_crlf(buf + pos, len - pos);
    if (crlf < 0) return -1;

    /* Parse method */
    int i = 0;
    while (pos < len && buf[pos] != ' ' && i < HTTP_MAX_METHOD - 1)
        req->method[i++] = buf[pos++];
    req->method[i] = '\0';
    if (pos >= len || buf[pos] != ' ') return -1;
    pos++;  /* skip space */

    /* Parse path */
    i = 0;
    while (pos < len && buf[pos] != ' ' && buf[pos] != '\r' && i < HTTP_MAX_PATH - 1)
        req->path[i++] = buf[pos++];
    req->path[i] = '\0';

    /* Parse HTTP version (optional; missing → HTTP/0.9) */
    if (pos < len && buf[pos] == ' ') {
        pos++;  /* skip space */
        i = 0;
        while (pos < len && buf[pos] != '\r' && i < HTTP_MAX_VERSION - 1)
            req->http_version[i++] = buf[pos++];
        req->http_version[i] = '\0';
    }

    /* skip \r\n */
    if (pos + 1 < len && buf[pos] == '\r' && buf[pos+1] == '\n')
        pos += 2;
    else
        return -1;

    /* ── Headers: "Name: Value\r\n" ... "\r\n" ───────────────────────── */
    while (pos < len && req->header_count < HTTP_MAX_HEADERS) {
        /* Empty line signals end of headers */
        if (pos + 1 < len && buf[pos] == '\r' && buf[pos+1] == '\n') {
            pos += 2;
            break;
        }

        /* Header name: up to ':' */
        i = 0;
        char name[HTTP_MAX_HDR_NAME];
        while (pos < len && buf[pos] != ':' && buf[pos] != '\r' && i < (int)sizeof(name)-1)
            name[i++] = buf[pos++];
        name[i] = '\0';

        if (pos >= len || buf[pos] != ':') {
            /* Malformed header line — skip to next CRLF */
            while (pos + 1 < len && !(buf[pos] == '\r' && buf[pos+1] == '\n'))
                pos++;
            pos += 2;
            continue;
        }
        pos++;  /* skip ':' */

        /* Skip optional whitespace after ':' */
        while (pos < len && buf[pos] == ' ') pos++;

        /* Header value: up to \r\n */
        i = 0;
        char value[HTTP_MAX_HDR_VAL];
        while (pos < len && buf[pos] != '\r' && i < (int)sizeof(value)-1)
            value[i++] = buf[pos++];
        value[i] = '\0';

        /* Skip \r\n */
        if (pos + 1 < len && buf[pos] == '\r' && buf[pos+1] == '\n')
            pos += 2;

        /* Store header */
        int h = req->header_count++;
        strncpy(req->headers[h][0], name,  HTTP_MAX_HDR_VAL - 1);
        strncpy(req->headers[h][1], value, HTTP_MAX_HDR_VAL - 1);
    }

    /* ── Body (if Content-Length header present) ─────────────────────── */
    const char *cl = http_find_header(req, "Content-Length");
    if (cl) {
        int body_len = atoi(cl);
        if (body_len > HTTP_MAX_BODY) body_len = HTTP_MAX_BODY;
        if (pos + body_len <= len) {
            memcpy(req->body, buf + pos, body_len);
            req->body_len = body_len;
        }
    }

    return 0;
}

/* ── build_response ──────────────────────────────────────────────────────── */

int build_response(int status_code, const char *status_text,
                   const char *content_type,
                   const char *body, int body_len,
                   char *out_buf, int out_size) {
    /*
     * Build the HTTP Date header (required by HTTP/1.1).
     * WHY required?  Caches and proxies use the Date header to calculate
     * how stale a cached response is.
     */
    char date_buf[64];
    time_t now = time(NULL);
    struct tm *gmt = gmtime(&now);
    strftime(date_buf, sizeof(date_buf), "%a, %d %b %Y %H:%M:%S GMT", gmt);

    int n = snprintf(out_buf, out_size,
        "HTTP/1.1 %d %s\r\n"
        "Date: %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "\r\n",
        status_code, status_text,
        date_buf,
        content_type ? content_type : "text/plain",
        body_len);

    if (n < 0 || n >= out_size) return -1;

    /* Append body if provided */
    if (body && body_len > 0) {
        if (n + body_len >= out_size) return -1;
        memcpy(out_buf + n, body, body_len);
        n += body_len;
    }

    return n;
}

/* ── http_find_header ────────────────────────────────────────────────────── */

const char *http_find_header(const HttpRequest *req, const char *name) {
    for (int i = 0; i < req->header_count; i++) {
        if (str_ci_eq(req->headers[i][0], name))
            return req->headers[i][1];
    }
    return NULL;
}
