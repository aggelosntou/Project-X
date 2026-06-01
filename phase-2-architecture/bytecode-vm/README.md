# Lox-mini Bytecode VM

A complete stack-based bytecode virtual machine implementing a subset of the Lox language (from "Crafting Interpreters" by Robert Nystrom). Written in C++17.

## Features

- **Language**: Integers, booleans, strings, nil, arithmetic, comparison, logical operators, variables, print, if/else, while
- **Single-pass compilation**: Pratt parser emits bytecode directly — no AST
- **Stack-based VM**: Fast instruction dispatch via switch statement
- **REPL + file execution**

## Building

```bash
make
```

## Running Programs

```bash
make run-fib        # Fibonacci numbers
make run-fizzbuzz   # FizzBuzz
make run-counter    # Global variables, sum 1..100

./lox_mini          # REPL
./lox_mini programs/fib.lox
```

## Language Syntax

```lox
// Variable declaration
var x = 10;
var name = "hello";
var flag = true;

// Arithmetic
var result = (x * 3 + 5) / 2 % 7;

// Comparison
if (x > 5 and name != "world") {
    print result;
}

// While loop
var i = 0;
while (i < 10) {
    print i;
    i = i + 1;
}
```

## Architecture

```
Source text
    │
    ▼
Scanner (scanner.cpp)         → Token stream
    │
    ▼
Compiler (compiler.cpp)       → Bytecode (Pratt parser, no AST)
    │                            Constant pool
    ▼
VM (vm.cpp)                   → Eval loop (switch on opcode)
    │
    ▼
Output
```

## Opcodes

See `src/opcodes.hpp` for the full opcode list. Key instructions:

| Opcode | Description |
|---|---|
| `OP_CONST <idx>` | Push constant from pool |
| `OP_ADD/SUB/MUL/DIV/MOD` | Binary arithmetic |
| `OP_JUMP <hi> <lo>` | Unconditional jump |
| `OP_JUMP_IF_FALSE <hi> <lo>` | Conditional jump (for if/while) |
| `OP_LOOP <hi> <lo>` | Backward jump (loop body) |
| `OP_DEFINE/GET/SET_GLOBAL <idx>` | Variable operations |
