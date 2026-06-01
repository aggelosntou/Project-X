# LEARNINGS — tcp-server-rust

## 1. Rust ownership model: why Arc is needed across threads

Rust's ownership system guarantees at compile time that every value has exactly one owner. When that owner goes out of scope the value is dropped. This eliminates use-after-free and double-free bugs without a garbage collector.

The problem with threads: a spawned thread may outlive the function that created it. If the thread holds a reference to a local variable, that variable could be freed while the thread is still running — a classic use-after-free. Rust rejects this at compile time through the `'static` bound on `thread::spawn`.

**`Arc<T>` (Atomically Reference Counted)** solves this by moving the value onto the heap and handing out handles to it. `Arc::clone()` gives a new handle; the value lives as long as at least one handle exists. Each clone/drop atomically increments/decrements a reference count — "atomic" means the operation is a single, indivisible CPU instruction, safe to call from multiple threads simultaneously.

```rust
let counter = Arc::new(0u64);         // value on the heap, refcount = 1
let counter2 = Arc::clone(&counter);  // refcount = 2
thread::spawn(move || {
    // counter2 is moved into this thread
    // the value lives until BOTH counter and counter2 are dropped
    println!("{}", *counter2);
});
// counter still valid here; counter2 was moved
```

## 2. Arc<Mutex<T>>: the "shared XOR mutable" rule

Rust enforces a fundamental invariant:

> You may have **many shared (immutable) references** OR **one exclusive (mutable) reference**, but never both at the same time.

This is checked at compile time for single-threaded code (`&T` vs. `&mut T`). For concurrent code the same rule applies but at runtime:

- `Arc<T>` lets many threads share ownership. If `T` were a plain `u64`, all threads would read it freely, but there is no way to mutate it — immutable sharing is safe but useless here.
- **`Mutex<T>`** wraps `T` and enforces the rule at runtime: `lock()` blocks until no other thread holds the mutex, then returns a `MutexGuard<T>` that gives `&mut T` access. When the guard is dropped the lock is released. Only one `MutexGuard` exists at a time → only one mutable access at a time.

Together:

```rust
let shared: Arc<Mutex<u64>> = Arc::new(Mutex::new(0));

// In thread A:
let mut guard = shared.lock().unwrap();  // blocks if thread B holds it
*guard += 1;                             // exclusive mutable access
// guard dropped here → lock released

// In thread B: same pattern
```

Why not just `Arc<u64>` with interior mutation tricks? You can — using `Arc<AtomicU64>` for simple integers, which is what we do for the `running` flag. But for a complex struct `Mutex` is the general solution.

## 3. Why the compiler rejects a raw u64 shared across threads

Two marker traits govern thread safety in Rust:

- **`Send`**: a type is `Send` if it is safe to *transfer* to another thread. Most types are `Send`. A raw pointer is not (it carries no ownership information).
- **`Sync`**: a type is `Sync` if it is safe to *share a reference* (`&T`) with another thread. `T: Sync` implies `&T: Send`.

`u64` is `Send` (you can move it into a thread) but a `&mut u64` in two threads simultaneously would be a data race. Rust prevents this: `thread::spawn` requires `F: Send + 'static` for the closure. If the closure captures `&mut u64`, that reference is not `Send` (mutable reference sharing is unsafe), so it fails to compile:

```
error[E0277]: `*mut u64` cannot be shared between threads safely
```

`Arc<Mutex<u64>>` IS `Send + Sync` because the mutex guarantees exclusive access. The compiler's enforcement here catches concurrency bugs that in C would only manifest as intermittent runtime corruption.

## 4. Comparison: Rust threads vs. C thread pool — compile-time guarantees

| Property | Rust (`Arc<Mutex<T>>`) | C (`pthread_mutex_t`) |
|---|---|---|
| Forgetting to lock | Compile error (can't get `&mut T` without `lock()`) | Silent data race at runtime |
| Double-free / use-after-free | Prevented by ownership system | Requires valgrind / asan |
| Dead thread referencing freed memory | 'static bound prevents it | Common source of CVEs |
| Mutex poisoning detection | `lock()` returns `Err` if holder panicked | No equivalent; state undefined |
| Send/Sync checking | Compiler verifies every type crossing thread boundaries | Programmer's responsibility |

Rust's approach trades programmer freedom for safety guarantees enforced without runtime overhead (the checks happen at zero cost — compile time). A C thread pool can be written correctly, but the language gives no help: every invariant must be maintained by convention and discipline. In a large codebase this is a significant source of bugs.

The key insight: in Rust, if the server compiles, it is free of data races. In C, a code review and runtime testing under load are needed to gain the same confidence — and even then races can be rare enough to escape testing.
