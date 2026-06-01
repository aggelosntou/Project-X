# Learnings — hello-systems

## 1. The Four Stages of Compilation

Most people think of compilation as a single step. It is actually a pipeline
of four distinct programs, each transforming the source into a form closer
to machine code.

### Stage 1: Preprocessor (`cpp`)

The preprocessor is a pure **text substitution** engine. It never "understands"
C — it only processes lines beginning with `#`.

- `#include <stdio.h>` → copy-pastes the entire contents of `stdio.h` here
- `#define MAX 100` → replaces every occurrence of `MAX` with the literal `100`
- `#ifdef DEBUG ... #endif` → conditionally includes or excludes code blocks
- `__FILE__`, `__LINE__`, `__DATE__` are special macros the preprocessor fills in

Run `gcc -E src/main.c` to see the output. It's usually thousands of lines
because `stdio.h` drags in many other headers.

### Stage 2: Compiler (`cc1`)

The compiler proper translates the preprocessed C source into **assembly
language** — human-readable mnemonics for machine instructions
(`mov rax, 1`, `call printf`, etc.).

This is where:
- Type checking happens (the compiler understands C's type system)
- Optimisation happens (`-O2` enables ~200 separate optimisations)
- Warnings are generated (`-Wall -Wextra`)

Run `gcc -S src/main.c -o main.s` and open `main.s` to read the assembly.

### Stage 3: Assembler (`as`)

The assembler translates assembly text into **binary machine code** stored in
an **object file** (`.o`). An object file contains:

- `.text` section: the actual machine instructions for your functions
- `.data` section: your initialised global variables
- `.bss`  section: uninitialised global variables (just a size, no bytes stored)
- A **symbol table**: a list of function/variable names and their addresses

At this point, references to external symbols (like `printf`) are still
unresolved — there's just a placeholder that says "fill in the address of
`printf` later."

### Stage 4: Linker (`ld`)

The linker combines multiple `.o` files and library archives (`.a`, `.so`)
into a single executable. It:

1. Merges all the `.text`, `.data`, `.bss` sections into one address space
2. Resolves every external symbol reference to a concrete address
3. Writes the final ELF (Linux) or Mach-O (macOS) binary

When you use `printf`, the linker finds its code in the C standard library
(`libc`) and records how to reach it.

---

## 2. What a Makefile Is and Why It Exists

A Makefile is a **dependency graph** + **build recipes**.

The key insight: Make compares file **modification timestamps**. If `main.c`
was modified after `hello` was last built, Make reruns the compiler.
Otherwise it says "Nothing to be done" and skips the work.

For a project with 1000 source files, this means editing one file only
recompiles that file plus whatever depends on it — not everything.

The alternative is a shell script that always recompiles everything.
For large projects (Linux kernel: ~50,000 files) a full build takes hours.
Make brings it down to seconds for small changes.

---

## 3. What `-Wall -Wextra` Catch

These flags enable GCC/Clang's **static analysis** of your code as it compiles.

`-Wall` catches the most dangerous common errors:
- Using a variable before initialising it
- `printf` format string mismatches (`printf("%d", "hello")`)
- Implicit function declarations (calling a function without `#include`)
- Comparing signed and unsigned integers
- Unreachable code

`-Wextra` catches subtler issues:
- Function parameters that are never used
- Missing `break` in a switch statement (possible fallthrough bug)
- Empty loop bodies (`while(x--);` — was the `;` intentional?)
- Comparing a pointer to 0 directly

`-Werror` promotes every warning to a **compile error**. This is used in CI
pipelines to ensure no new warnings are ever introduced.

`-pedantic` rejects any GCC-specific extensions and demands strictly
conforming standard C. This improves portability.

---

## 4. What Undefined Behaviour Is, and Why C Has It

**Undefined behaviour (UB)** means the C standard explicitly says "the
implementation may do anything" for that operation. There is no guaranteed
result — not even a crash.

Common examples:
- Signed integer overflow (`INT_MAX + 1`)
- Out-of-bounds array access (`a[10]` when `a` has 5 elements)
- Dereferencing a NULL or dangling pointer
- Shifting by >= the bit width of the type (`x >> 32` for a 32-bit int)
- Modifying a string literal

### Why does C have UB instead of defined behaviour?

Because **defined behaviour has a cost**.

If signed overflow were defined to wrap (like unsigned integers), the compiler
would have to insert a mask instruction after every addition, or give up on
the optimisation that assumes `x + 1 > x` is always true.

If array bounds were always checked, the compiler would have to insert a
comparison before every array access — a significant overhead for
numerical code.

C trusts the programmer to never invoke UB, in exchange for generating the
fastest possible machine code for correct programs.

### The practical lesson

UB is the #1 source of security vulnerabilities in C programs. Buffer
overflows, use-after-free, and integer overflows are all forms of UB.

Tools that help:
- `-fsanitize=address` (AddressSanitizer) — detects memory bugs at runtime
- `-fsanitize=undefined` (UBSan) — detects integer overflow, null deref, etc.
- `valgrind` — slower but more thorough memory checker
- Static analysers: `clang-tidy`, `cppcheck`, `Coverity`
