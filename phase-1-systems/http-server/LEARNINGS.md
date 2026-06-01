# LEARNINGS — http-server

## TCP and sockets

TCP (Transmission Control Protocol) provides a reliable, ordered, byte-stream
abstraction over the unreliable IP packet layer.  The OS handles:
- Splitting your data into packets
- Retransmitting lost packets
- Ordering out-of-order packets
- Flow control (don't overwhelm the receiver)
- Congestion control (don't overwhelm the network)

A **socket** is a file descriptor representing one end of a network connection.
On Unix, it behaves like any other file descriptor: you can `read()`, `write()`,
`close()` it.

### Server socket lifecycle

```
socket(AF_INET, SOCK_STREAM, 0)
  → creates an endpoint (just an fd, no address assigned)

bind(fd, {AF_INET, port, INADDR_ANY}, ...)
  → assigns address+port to the socket

listen(fd, backlog)
  → marks socket as passive; kernel starts accepting SYN packets
  → backlog = max length of the queue of not-yet-accepted connections

accept(fd, &client_addr, &addr_len)
  → BLOCKS until a client completes the 3-way TCP handshake
  → returns a NEW fd specifically for that connection
  → original fd keeps listening for more connections
```

### Why separate listen and connection fds?

`accept()` returns a new fd because the listening socket needs to stay open
for future connections.  The new fd has its own receive/send buffers and
represents exactly one TCP connection.

## poll() — I/O multiplexing

A **blocking** server calls `recv()` on a client socket and waits.  During
that wait, no other client can be served.  Two solutions:

1. **Thread per connection**: fork or thread for each client.  Overhead per
   connection: ~8 KB stack + OS scheduling.  Works to ~10,000 connections.

2. **I/O multiplexing**: a single thread watches many fds simultaneously,
   waking up only when one is ready.  Works to millions of connections
   (the "C10K problem").

`poll()` takes an array of `struct pollfd {fd, events, revents}` and a
timeout.  It returns when at least one fd matches:

```c
struct pollfd fds[2];
fds[0].fd = listen_fd; fds[0].events = POLLIN;
fds[1].fd = client_fd; fds[1].events = POLLIN;

poll(fds, 2, -1);   // wait forever

if (fds[0].revents & POLLIN)  // new connection
if (fds[1].revents & POLLIN)  // client sent data
```

`POLLIN`  = data available for reading (won't block)
`POLLOUT` = space available for writing (won't block)
`POLLERR` = error on the fd
`POLLHUP` = the other end closed the connection

### poll() vs select() vs epoll() vs kqueue()

| API | Platform | Max fds | Complexity | Notes |
|-----|---------|---------|-----------|-------|
| `select()` | POSIX | FD_SETSIZE (~1024) | O(n) | Oldest; severely limited |
| `poll()` | POSIX | Unlimited | O(n) | Good portability |
| `epoll()` | Linux only | Millions | O(1) | Production choice on Linux |
| `kqueue()` | macOS/BSD | Millions | O(1) | Production choice on macOS |

We use `poll()` for portability.  For a production macOS server, use `kqueue()`.

## HTTP/1.1

HTTP is a text-based request/response protocol over TCP.

### Request format

```
GET /hello HTTP/1.1\r\n
Host: localhost:8080\r\n
Accept: application/json\r\n
\r\n
```

The double `\r\n` terminates the headers.  A body follows if `Content-Length`
or `Transfer-Encoding` is present.

### Response format

```
HTTP/1.1 200 OK\r\n
Date: Mon, 01 Jun 2026 12:00:00 GMT\r\n
Content-Type: application/json\r\n
Content-Length: 42\r\n
Connection: close\r\n
\r\n
{"message":"hello","server":"c-http-server"}
```

### Why `Content-Length`?

HTTP/1.1 uses persistent connections by default (`Connection: keep-alive`).
Without `Content-Length` (or chunked transfer encoding), the client can't know
where one response ends and the next begins.  We send `Connection: close` and
include the content length to keep things simple.

### Status codes

| Code | Meaning |
|------|---------|
| 200 | OK — successful |
| 400 | Bad Request — malformed syntax |
| 404 | Not Found — no such resource |
| 405 | Method Not Allowed |
| 500 | Internal Server Error |

## Non-blocking sockets

With `O_NONBLOCK`, `recv()` and `send()` return immediately:
- If data is available: return the data
- If no data: return -1 with `errno == EAGAIN` (try again)

This is critical with `poll()`: after `poll()` says a socket is readable, a
single `recv()` may not return all the data (TCP is a stream, not a message
protocol).  In a production server you'd loop until `EAGAIN`.  We simplify
by assuming a complete HTTP request arrives in one `recv()` call.
