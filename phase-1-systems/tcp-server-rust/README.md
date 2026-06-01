# tcp-server-rust

A multi-threaded TCP echo server written in Rust using only the standard library. Each client connection is handled in its own OS thread. Lines sent by the client are echoed back prefixed with their line number within that connection.

## Build

```bash
cargo build --release
```

The binary is placed at `target/release/tcp-server`.

## Run

```bash
cargo run --release
```

You will see:
```
[server] listening on 127.0.0.1:7878
[server] type "quit" and press Enter to shut down
```

## Test with netcat

Open a second terminal:

```bash
nc 127.0.0.1 7878
```

Type lines and observe them echoed back with a line counter:

```
hello
1: hello
world
2: world
foo bar baz
3: foo bar baz
```

Disconnect with Ctrl+C. The server prints the number of lines that connection exchanged.

Open multiple terminals connecting simultaneously to see the per-connection isolation: line counters restart at 1 for each new client.

## Observe

- **Line numbering**: each connection's line counter is independent.
- **Connection count**: the server prints a global counter each time a new client is accepted.
- **Graceful shutdown**: type `quit` + Enter in the server terminal. The server stops accepting new connections; existing connections finish naturally.

## Key concepts demonstrated

| Concept | Location |
|---------|----------|
| `Arc<Mutex<u64>>` shared counter | `main()` and `handle_connection()` |
| Why raw `u64` across threads is rejected | Comment block at top of `main.rs` |
| Per-thread connection handler | `thread::spawn(move \|\| { ... })` |
| `AtomicBool` for shutdown flag | `running` variable in `main()` |
| Stdin-based shutdown | background thread reading `stdin` |
