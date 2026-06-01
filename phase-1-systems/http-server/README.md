# http-server

An HTTP/1.1 server written in C using raw sockets and `poll()` for I/O
multiplexing.  No frameworks, no external libraries.

## Features

- Handles multiple concurrent connections in a single thread via `poll()`
- Non-blocking sockets
- Routes: `GET /`, `GET /hello`, `GET /stats`
- Proper HTTP/1.1 headers: `Content-Type`, `Content-Length`, `Date`
- Request logging: method, path, status, response time

## How to build

```bash
make all
```

## How to run

```bash
./bin/server          # listens on port 8080
./bin/server 9090     # listens on port 9090
```

Then in another terminal:

```bash
curl http://localhost:8080/
curl http://localhost:8080/hello
curl http://localhost:8080/stats
```

Or open `http://localhost:8080` in a browser.

## Files

| File | Purpose |
|------|---------|
| `src/http_parser.h` | HttpRequest struct + parser API |
| `src/http_parser.c` | HTTP/1.1 request parser + response builder |
| `src/server.c` | TCP server with poll() multiplexing + routing |
| `Makefile` | Build system |
| `README.md` | This file |
| `LEARNINGS.md` | TCP, sockets, poll(), HTTP/1.1 explained |
