/*
 * server.c — HTTP/1.1 server using poll() for I/O multiplexing.
 *
 * ── NETWORKING CONCEPTS ────────────────────────────────────────────────────
 *
 * TCP AND SOCKETS
 *   A socket is a communication endpoint identified by (IP, port, protocol).
 *   TCP provides reliable, ordered byte streams — the OS handles
 *   retransmission, ordering, and flow control.
 *
 *   Server lifecycle:
 *     socket()  → create endpoint (just an fd, not yet bound)
 *     bind()    → assign an address (IP + port)
 *     listen()  → mark socket as passive (waiting for connections)
 *     accept()  → block until a client connects, return a NEW fd for that conn
 *
 *   Client sends "SYN" → server responds "SYN+ACK" → client sends "ACK".
 *   This 3-way handshake completes before accept() returns.
 *
 * POLL() VS BLOCKING I/O
 *   A naive server calls accept(), then recv() on the new socket.
 *   Problem: recv() BLOCKS until data arrives.  While blocked on one client,
 *   other clients cannot connect or send data.
 *
 *   poll() takes a list of file descriptors and a timeout.  It returns when
 *   ANY of the fds are "ready" (readable/writable without blocking).  This
 *   lets us handle N connections in a single thread — the "reactor" pattern.
 *
 *   On macOS, kqueue() is more efficient for large numbers of fds, but poll()
 *   is portable and sufficient for teaching purposes.
 *
 * NON-BLOCKING SOCKETS
 *   Setting O_NONBLOCK on a socket means read/write return EAGAIN instead of
 *   blocking when no data is available.  Combined with poll(), this is the
 *   foundation of async I/O.
 */

#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE

#include "http_parser.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <poll.h>

/* ── Configuration ───────────────────────────────────────────────────────── */

#define DEFAULT_PORT      8080
#define MAX_CONNECTIONS   64
#define RECV_BUF_SIZE     8192
#define SEND_BUF_SIZE     65536
#define BACKLOG           128    /* listen queue depth */

/* ── Server statistics ───────────────────────────────────────────────────── */

typedef struct {
    long total_requests;
    long total_bytes_sent;
    double total_response_time_ms;
    long requests_200;
    long requests_404;
} ServerStats;

static ServerStats g_stats = {0};

/* ── Per-connection state ─────────────────────────────────────────────────── */

typedef struct {
    int     fd;
    char    recv_buf[RECV_BUF_SIZE];
    int     recv_len;
    char    send_buf[SEND_BUF_SIZE];
    int     send_len;
    int     send_offset;
    struct timespec connect_time;
} Connection;

/* ── Socket helpers ──────────────────────────────────────────────────────── */

static int set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

static int create_listen_socket(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); return -1; }

    /*
     * SO_REUSEADDR — allow binding to a port that is in TIME_WAIT.
     *
     * WHY needed?  When a TCP connection closes, the port stays in TIME_WAIT
     * for ~2×MSL (60–120 seconds) to absorb delayed packets.  Without this
     * option, restarting the server immediately fails with "Address already
     * in use".
     */
    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {
        .sin_family      = AF_INET,
        .sin_addr.s_addr = INADDR_ANY,
        .sin_port        = htons((uint16_t)port)
        /*
         * htons = "host to network short"
         * Network protocols use big-endian byte order.  x86/ARM are
         * little-endian.  htons swaps the bytes so the port number is
         * transmitted correctly.
         */
    };

    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(fd); return -1;
    }

    if (listen(fd, BACKLOG) < 0) {
        perror("listen"); close(fd); return -1;
    }

    set_nonblocking(fd);
    return fd;
}

/* ── Route handlers ──────────────────────────────────────────────────────── */

static int route_root(char *out, int out_size) {
    const char *html =
        "<!DOCTYPE html>\n"
        "<html><head><title>C HTTP Server</title></head>\n"
        "<body>\n"
        "<h1>Hello from our C HTTP server!</h1>\n"
        "<p>This server was written from scratch in C using raw sockets and poll().</p>\n"
        "<ul>\n"
        "  <li><a href=\"/hello\">/hello</a> — JSON greeting</li>\n"
        "  <li><a href=\"/stats\">/stats</a> — server statistics</li>\n"
        "</ul>\n"
        "</body></html>\n";
    return build_response(200, "OK", "text/html; charset=utf-8",
                          html, (int)strlen(html), out, out_size);
}

static int route_hello(char *out, int out_size) {
    const char *json = "{\"message\":\"hello\",\"server\":\"c-http-server\",\"version\":\"1.0\"}\n";
    return build_response(200, "OK", "application/json",
                          json, (int)strlen(json), out, out_size);
}

static int route_stats(char *out, int out_size) {
    char body[512];
    double avg_rt = g_stats.total_requests > 0
                    ? g_stats.total_response_time_ms / g_stats.total_requests
                    : 0.0;
    snprintf(body, sizeof(body),
        "{\n"
        "  \"total_requests\": %ld,\n"
        "  \"total_bytes_sent\": %ld,\n"
        "  \"requests_200\": %ld,\n"
        "  \"requests_404\": %ld,\n"
        "  \"avg_response_time_ms\": %.3f\n"
        "}\n",
        g_stats.total_requests,
        g_stats.total_bytes_sent,
        g_stats.requests_200,
        g_stats.requests_404,
        avg_rt);
    return build_response(200, "OK", "application/json",
                          body, (int)strlen(body), out, out_size);
}

static int route_not_found(char *out, int out_size) {
    const char *body = "{\"error\":\"not found\"}\n";
    return build_response(404, "Not Found", "application/json",
                          body, (int)strlen(body), out, out_size);
}

/* ── Request dispatch ────────────────────────────────────────────────────── */

static double timespec_diff_ms(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1000.0
         + (end->tv_nsec - start->tv_nsec) / 1e6;
}

static void handle_request(Connection *conn) {
    HttpRequest req;
    if (parse_request(conn->recv_buf, conn->recv_len, &req) < 0) {
        conn->send_len = build_response(400, "Bad Request", "text/plain",
                                        "Bad Request", 11,
                                        conn->send_buf, SEND_BUF_SIZE);
        return;
    }

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    int len = 0;
    int status = 200;

    if (strcmp(req.method, "GET") == 0) {
        if (strcmp(req.path, "/") == 0 || strcmp(req.path, "/index.html") == 0) {
            len = route_root(conn->send_buf, SEND_BUF_SIZE);
        } else if (strcmp(req.path, "/hello") == 0) {
            len = route_hello(conn->send_buf, SEND_BUF_SIZE);
        } else if (strcmp(req.path, "/stats") == 0) {
            len = route_stats(conn->send_buf, SEND_BUF_SIZE);
        } else {
            len    = route_not_found(conn->send_buf, SEND_BUF_SIZE);
            status = 404;
        }
    } else {
        /* Method not allowed */
        const char *body = "{\"error\":\"method not allowed\"}\n";
        len    = build_response(405, "Method Not Allowed", "application/json",
                                body, (int)strlen(body),
                                conn->send_buf, SEND_BUF_SIZE);
        status = 405;
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double rt_ms = timespec_diff_ms(&t_start, &t_end);

    if (len > 0) conn->send_len = len;

    /* Update statistics */
    g_stats.total_requests++;
    g_stats.total_bytes_sent    += len;
    g_stats.total_response_time_ms += rt_ms;
    if (status == 200) g_stats.requests_200++;
    if (status == 404) g_stats.requests_404++;

    /* Log the request */
    printf("[%ld] %s %s → %d  (%.2f ms, %d bytes)\n",
           (long)g_stats.total_requests,
           req.method, req.path, status, rt_ms, len);
}

/* ── Main server loop ────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    int port = DEFAULT_PORT;
    if (argc > 1) port = atoi(argv[1]);

    int listen_fd = create_listen_socket(port);
    if (listen_fd < 0) return 1;

    printf("HTTP server listening on http://localhost:%d\n", port);
    printf("Routes: GET /   GET /hello   GET /stats\n\n");

    /*
     * poll() file descriptor array.
     *
     * struct pollfd:
     *   fd      — the file descriptor to watch
     *   events  — what events we're interested in (POLLIN = readable)
     *   revents — what events actually occurred (filled by poll())
     *
     * We keep the listen socket at index 0, client connections at 1..N.
     */
    struct pollfd fds[MAX_CONNECTIONS + 1];
    Connection    conns[MAX_CONNECTIONS];
    int           num_clients = 0;

    /* Slot 0: the listening socket */
    fds[0].fd     = listen_fd;
    fds[0].events = POLLIN;

    /* Initialise client slots */
    for (int i = 0; i < MAX_CONNECTIONS; i++) {
        fds[i+1].fd = -1;  /* -1 means "ignore this slot" */
        conns[i].fd  = -1;
    }

    while (1) {
        /*
         * poll() blocks until at least one fd is ready, or timeout (-1 = forever).
         *
         * WHY not select()?  select() uses fd_set bitmasks with a hard limit
         * of FD_SETSIZE (typically 1024) fds.  poll() takes an array and has
         * no hard limit.  On macOS, kqueue() is the production choice, but
         * poll() is more portable and sufficient here.
         */
        int ready = poll(fds, (nfds_t)(num_clients + 1), -1);
        if (ready < 0) {
            if (errno == EINTR) continue;  /* interrupted by signal, retry */
            perror("poll");
            break;
        }

        /* ── Check for new connections on the listen socket ─────────── */
        if (fds[0].revents & POLLIN) {
            struct sockaddr_in client_addr;
            socklen_t          addr_len = sizeof(client_addr);
            int client_fd = accept(listen_fd,
                                   (struct sockaddr *)&client_addr,
                                   &addr_len);
            if (client_fd >= 0) {
                set_nonblocking(client_fd);

                /* Find an empty slot */
                int slot = -1;
                for (int i = 0; i < MAX_CONNECTIONS; i++) {
                    if (conns[i].fd < 0) { slot = i; break; }
                }

                if (slot < 0) {
                    /* No slots available — reject connection */
                    fprintf(stderr, "max connections reached, dropping client\n");
                    close(client_fd);
                } else {
                    fds[slot+1].fd      = client_fd;
                    fds[slot+1].events  = POLLIN;
                    fds[slot+1].revents = 0;

                    memset(&conns[slot], 0, sizeof(Connection));
                    conns[slot].fd = client_fd;
                    clock_gettime(CLOCK_MONOTONIC, &conns[slot].connect_time);
                    num_clients++;

                    printf("New connection from %s:%d (fd=%d, slot=%d)\n",
                           inet_ntoa(client_addr.sin_addr),
                           ntohs(client_addr.sin_port),
                           client_fd, slot);
                }
            }
        }

        /* ── Service existing connections ────────────────────────────── */
        for (int i = 0; i < MAX_CONNECTIONS; i++) {
            if (fds[i+1].fd < 0) continue;

            Connection *conn = &conns[i];

            /* ── Readable: data from client ──────────────────────────── */
            if (fds[i+1].revents & POLLIN) {
                int n = (int)recv(conn->fd,
                                  conn->recv_buf + conn->recv_len,
                                  RECV_BUF_SIZE - conn->recv_len - 1,
                                  0);
                if (n <= 0) {
                    /* Connection closed or error */
                    goto close_conn;
                }

                conn->recv_len += n;
                conn->recv_buf[conn->recv_len] = '\0';

                /*
                 * Detect end of HTTP headers (\r\n\r\n).
                 * A production server would also handle large bodies and
                 * chunked transfer encoding.
                 */
                if (strstr(conn->recv_buf, "\r\n\r\n")) {
                    handle_request(conn);
                    conn->send_offset = 0;
                    /* Switch to watching for writability */
                    fds[i+1].events = POLLOUT;
                }
            }

            /* ── Writable: send the response ─────────────────────────── */
            if (fds[i+1].revents & POLLOUT) {
                int remaining = conn->send_len - conn->send_offset;
                if (remaining > 0) {
                    int sent = (int)send(conn->fd,
                                         conn->send_buf + conn->send_offset,
                                         remaining, 0);
                    if (sent < 0 && errno != EAGAIN) goto close_conn;
                    if (sent > 0)  conn->send_offset += sent;
                }

                /* All sent → close connection (HTTP/1.0-style) */
                if (conn->send_offset >= conn->send_len) {
                    goto close_conn;
                }
            }

            /* ── Error or hangup ─────────────────────────────────────── */
            if (fds[i+1].revents & (POLLERR | POLLHUP)) {
                goto close_conn;
            }

            continue;

        close_conn:
            close(conn->fd);
            conn->fd    = -1;
            fds[i+1].fd = -1;
            num_clients--;
        }
    }

    close(listen_fd);
    return 0;
}
