//! tcp-server — a simple TCP echo server
//!
//! Features:
//!   • Binds to 127.0.0.1:7878
//!   • Each incoming connection is handled in its own OS thread
//!   • Each line received is echoed back as "N: <line>" where N is the
//!     per-connection line counter
//!   • A global connection counter (Arc<Mutex<u64>>) is incremented and
//!     printed each time a new client connects
//!   • Type a line containing "quit" into the server's stdin to shut down
//!     gracefully: the listener stops accepting and all worker threads finish
//!     their current connections before the process exits
//!
//! Build:   cargo build --release
//! Run:     cargo run --release
//! Test:    nc 127.0.0.1 7878   (type lines, see them echoed with numbers)

// ---------------------------------------------------------------------------
// WHY Arc<Mutex<u64>> instead of a plain u64?
//
// Imagine this naive approach:
//
//   let mut counter: u64 = 0;           // lives on the stack of main()
//   for stream in listener.incoming() {
//       counter += 1;                    // modify before spawning
//       std::thread::spawn(|| {
//           println!("conn #{}", counter); // ERROR: cannot capture `counter`
//       });
//   }
//
// The compiler rejects this for TWO reasons:
//
//  1. `counter` is borrowed by the closure, but `thread::spawn` requires its
//     closure to be 'static (live forever), because the spawned thread may
//     outlive the current stack frame. A reference to a stack variable is NOT
//     'static.
//
//  2. Even if we tried to move `counter` into the closure, it would be
//     consumed by the first iteration — unavailable for the second.
//
//  3. Sharing a raw `u64` between threads is not `Sync` (i.e. it doesn't
//     implement the `Sync` marker trait). The compiler enforces this to
//     prevent data races at compile time — no runtime check needed.
//
// The correct solution:
//   • Arc<T> (Atomic Reference Counted pointer) lets multiple threads SHARE
//     ownership of the same heap allocation. Cloning an Arc bumps a counter
//     atomically; dropping it decrements it; the value is freed when the
//     count reaches zero.
//   • Mutex<T> provides exclusive mutable access. Only one thread can hold
//     the lock at a time. This satisfies Rust's "shared XOR mutable" rule:
//     you can have many readers OR one writer, never both simultaneously.
//
// Together, Arc<Mutex<u64>> is Send + Sync and compiles cleanly.
// ---------------------------------------------------------------------------

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

/// Handle a single client connection.
///
/// Reads lines from the client and echoes each one back prefixed with
/// its line number (1-indexed): "1: hello\n", "2: world\n", etc.
/// Returns when the client disconnects (EOF) or a read/write error occurs.
fn handle_connection(mut stream: TcpStream, conn_id: u64) {
    println!("[conn {}] accepted from {}", conn_id, stream.peer_addr().unwrap());

    // Wrap the stream in a buffered reader so we can read line-by-line.
    // We need a second clone for writing because BufReader takes ownership.
    let reader_stream = stream.try_clone().expect("failed to clone stream");
    let reader = BufReader::new(reader_stream);

    let mut line_number: u64 = 0;

    for line_result in reader.lines() {
        match line_result {
            Ok(line) => {
                line_number += 1;
                let response = format!("{}: {}\n", line_number, line);
                if stream.write_all(response.as_bytes()).is_err() {
                    break; // client disconnected while we were writing
                }
            }
            Err(_) => break, // EOF or read error — client disconnected
        }
    }

    println!(
        "[conn {}] closed after {} lines",
        conn_id, line_number
    );
}

fn main() {
    // -----------------------------------------------------------------------
    // Shared state
    // -----------------------------------------------------------------------

    // Connection counter shared between the accept loop and each handler thread.
    //
    // Arc::clone() produces a new "handle" to the same heap allocation; the
    // underlying u64 lives until all Arc handles are dropped.
    // Mutex::lock() returns a MutexGuard that auto-unlocks when dropped.
    let connection_count: Arc<Mutex<u64>> = Arc::new(Mutex::new(0));

    // Shutdown flag: an AtomicBool can be read/written from multiple threads
    // without a Mutex because the read-modify-write is a single CPU instruction.
    // We use Ordering::SeqCst for simplicity (the strongest memory order).
    let running: Arc<AtomicBool> = Arc::new(AtomicBool::new(true));

    // -----------------------------------------------------------------------
    // Graceful shutdown via stdin
    //
    // A second thread reads stdin. Typing a line containing "quit" sets
    // `running` to false, which causes the accept loop to stop.
    //
    // We clone the Arc before moving it into the thread closure.
    // -----------------------------------------------------------------------
    let running_for_stdin = Arc::clone(&running);
    thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            if let Ok(text) = line {
                if text.trim() == "quit" {
                    println!("[server] shutdown requested via stdin");
                    running_for_stdin.store(false, Ordering::SeqCst);
                    // Connect once to ourselves to unblock listener.accept()
                    // (the listener has no timeout so it blocks indefinitely).
                    let _ = std::net::TcpStream::connect("127.0.0.1:7878");
                    break;
                }
            }
        }
    });

    // -----------------------------------------------------------------------
    // Bind the listener
    // -----------------------------------------------------------------------
    let listener = TcpListener::bind("127.0.0.1:7878")
        .expect("failed to bind 127.0.0.1:7878 — is the port in use?");

    println!("[server] listening on 127.0.0.1:7878");
    println!("[server] type \"quit\" and press Enter to shut down");

    // -----------------------------------------------------------------------
    // Accept loop
    // -----------------------------------------------------------------------
    for stream_result in listener.incoming() {
        // Check shutdown flag FIRST so we exit cleanly even on the dummy
        // connection sent by the stdin thread.
        if !running.load(Ordering::SeqCst) {
            println!("[server] shutting down — no longer accepting connections");
            break;
        }

        match stream_result {
            Ok(stream) => {
                // Increment connection counter
                let conn_id = {
                    let mut count = connection_count
                        .lock()
                        .expect("connection_count mutex poisoned");
                    *count += 1;
                    *count
                };
                // Total connections so far
                println!(
                    "[server] new connection #{} (total accepted: {})",
                    conn_id, conn_id
                );

                // Spawn a thread to handle this connection.
                //
                // The closure must be 'static and Send.
                // `stream` is moved in (ownership transfer — no sharing needed).
                // `conn_id` is a plain u64 (Copy), so it's implicitly copied in.
                thread::spawn(move || {
                    handle_connection(stream, conn_id);
                });
            }
            Err(e) => {
                // This can happen if the OS denies the accept syscall (rare).
                eprintln!("[server] accept error: {}", e);
            }
        }
    }

    println!("[server] exited accept loop — waiting for active threads to finish");
    // Note: we don't join the worker threads here — they will finish when their
    // clients disconnect and the process will then exit. For a production server
    // you would maintain a list of JoinHandles and join them here.
}
