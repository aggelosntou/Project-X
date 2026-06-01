# Mini Compiler — Calc to x86-64 Assembly

A complete single-pass compiler that translates the "Calc" language to
x86-64 AT&T assembly, demonstrating all three classical compiler phases:
lexer, parser, and code generator.

## Requirements

- GCC (or Clang) with C11 support
- Linux or macOS x86-64
- On Linux: `gcc -no-pie` to link position-independent executables

## Build

```bash
make          # compiles the calc compiler itself
make test     # compile and run all three example programs
```

## The Calc Language

```
// Variables
var x = 10;
var y = x + 5;

// Assignment
x = x * 2;

// Arithmetic: + - * / %
// Comparison: == != < > <= >=
// Logical: && || !
// Control flow: if/else, while
// Output: print <expr>;

// Comments: // to end of line

var i = 1;
while (i <= 10) {
    if (i % 2 == 0) {
        print i;
    }
    i = i + 1;
}
```

## Usage

```bash
./calc programs/add.calc       # produces add.s
gcc add.s -o add_prog          # Linux (may need -no-pie)
./add_prog                     # → 42

CALC_DUMP_AST=1 ./calc programs/fizzbuzz.calc   # print the AST first
```

## Example Programs

| File                       | Expected Output          |
|----------------------------|--------------------------|
| `programs/add.calc`        | `42`                     |
| `programs/loop.calc`       | `5050`                   |
| `programs/fizzbuzz.calc`   | FizzBuzz 1..20 encoded   |
| `programs/expressions.calc`| Tests all operators      |

FizzBuzz encoding: `0` = FizzBuzz, `3` = Fizz, `5` = Buzz, otherwise `i`.

## Source Files

| File              | Purpose                                       |
|-------------------|-----------------------------------------------|
| `src/lexer.h/.c`  | Tokenise source text into Token stream        |
| `src/parser.h/.c` | Recursive-descent parser, builds AST          |
| `src/codegen.h/.c`| Walk AST, emit x86-64 AT&T assembly           |
| `src/main.c`      | Read file → lex → parse → codegen → write .s |

## Architecture Overview

```
source text
    │
    ▼  lexer.c
Token stream  (TOK_NUMBER 10, TOK_IDENT x, TOK_PLUS, ...)
    │
    ▼  parser.c
AST (Abstract Syntax Tree)
    │   NODE_PROGRAM
    │     NODE_VAR_DECL "x" → NODE_NUMBER 10
    │     NODE_PRINT → NODE_BINOP "+" NODE_IDENT"x" NODE_IDENT"y"
    │
    ▼  codegen.c
x86-64 AT&T assembly  (.s file)
    │
    ▼  gcc / as + ld
native binary
```

## Generated Assembly Example

For `var x = 10; var y = 32; print x + y;`:

```asm
    .section .rodata
.fmt_int:
    .string "%ld\n"

    .section .text
    .global main
main:
    pushq   %rbp
    movq    %rsp, %rbp
    subq    $16, %rsp          # room for 2 variables

    movq    $10, %rax
    movq    %rax, -8(%rbp)     # var x

    movq    $32, %rax
    movq    %rax, -16(%rbp)    # var y

    movq    -8(%rbp), %rax     # load x
    pushq   %rax
    movq    -16(%rbp), %rax    # load y
    popq    %rcx
    addq    %rcx, %rax         # x + y

    movq    %rax, %rsi
    leaq    .fmt_int(%rip), %rdi
    xorl    %eax, %eax
    call    printf             # print result

    movq    $0, %rax
    leave
    ret
```
