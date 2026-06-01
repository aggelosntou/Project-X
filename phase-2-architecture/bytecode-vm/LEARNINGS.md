# Bytecode VM — Learnings

## Why a Stack Machine?

A **stack machine** uses an implicit stack for all intermediate values. Instructions don't name source/destination registers — they just push and pop.

Example: computing `2 + 3 * 4`

```
OP_CONST 2    → stack: [2]
OP_CONST 3    → stack: [2, 3]
OP_CONST 4    → stack: [2, 3, 4]
OP_MUL        → stack: [2, 12]    (pop 3,4; push 12)
OP_ADD        → stack: [14]       (pop 2,12; push 14)
```

Advantages:
- No **register allocation** needed (the hardest part of a compiler)
- Compact bytecode (operands are always the top of stack)
- Simple interpreter loop

Disadvantages:
- More instructions than a register machine for the same computation
- Stack operations create data hazards (CPython addressed this with specializing adaptive interpreter in 3.11+)

Real examples of stack machines: JVM (Java bytecode), CPython (dis module), WebAssembly.

## What the Constant Pool Is

The **constant pool** is a list of literal values embedded in the bytecode. When the compiler sees `var x = 42.5`, it:
1. Adds `42.5` to the constant pool at index 0
2. Emits `OP_CONST 0` — "push the constant at index 0"

This is necessary because bytecode instructions have fixed size (1-2 bytes). A 64-bit double doesn't fit inline. The pool is the solution: store the value once, reference it by index.

In the JVM, the constant pool also stores: class names, method signatures, field names. It's the "string table" of the compiled class file.

## How the Pratt Parser Works

A **Pratt parser** (or top-down operator precedence parser) assigns each token type:
- A **prefix parse function**: how to parse it as the start of an expression
- An **infix parse function**: how to parse it after a left-hand side has been parsed
- A **precedence level**: determines when to stop recursing

The key function `parse_precedence(min_prec)`:
1. Call the prefix function for the current token
2. While the next token's precedence ≥ min_prec: call its infix function

This elegantly handles operator precedence without explicit grammar rules for each precedence level.

Example: parsing `1 + 2 * 3`
```
parse_precedence(PREC_ASSIGNMENT)
  prefix: number() → emit 1
  peek: '+' has prec PREC_TERM ≥ PREC_ASSIGNMENT → advance, call binary('+')
    parse_precedence(PREC_FACTOR)  [one above TERM — left-assoc]
      prefix: number() → emit 2
      peek: '*' has prec PREC_FACTOR ≥ PREC_FACTOR → advance, call binary('*')
        parse_precedence(PREC_UNARY)
          prefix: number() → emit 3
      emit OP_MUL
    return
  emit OP_ADD
```

Emitted bytecode: `CONST 1, CONST 2, CONST 3, MUL, ADD` — correct!

Pratt parsers are used in: Rust compiler (rustc), Go compiler, V8 JavaScript engine.

## What Bytecode Buys You Over Tree-Walking

A **tree-walking interpreter** builds an AST and evaluates it by recursively walking the tree. This is simple but slow:

1. **Indirect dispatch**: every node is a virtual function call or switch on node type
2. **Memory allocation**: every expression creates heap-allocated AST nodes
3. **Cache-hostile**: AST nodes are scattered across the heap; tree traversal has poor spatial locality

Bytecode compilation:
1. Compiles once to a compact byte array
2. The eval loop is a tight switch statement — the hot path stays in L1 instruction cache
3. Constants are pooled contiguously
4. No allocation during execution (values live on the stack)

CPython's speed improvement from 3.10 to 3.11 (25% faster) came from specialized bytecode instructions that reduce dynamic dispatch in the eval loop.

## JIT Concepts (Not Implemented)

A **JIT (Just-In-Time) compiler** turns bytecode into native machine code at runtime.

Approaches:
1. **Method JIT** (JVM HotSpot): compile entire methods to native code when they run "hot" (many times)
2. **Tracing JIT** (LuaJIT, PyPy): detect hot loops, record a trace of executed instructions, compile just that trace
3. **Copy-and-patch** (CPython 3.13): generate native code by copying templates and patching in constants

For our VM, a simple JIT would:
1. Count how many times each instruction runs
2. When a loop body exceeds a threshold, compile it to x86-64 machine code
3. Jump to the native code instead of interpreting bytecode

The speedup from JIT: typically 10-100× over pure interpretation for compute-heavy code. This is how JavaScript engines (V8, SpiderMonkey) run so fast.

## Bytecode Format

Our bytecode uses 1-byte opcodes with optional 1-byte or 2-byte operands:

```
OP_CONST    <1-byte index>           2 bytes
OP_ADD                               1 byte
OP_JUMP     <hi-byte> <lo-byte>      3 bytes (16-bit offset)
OP_LOOP     <hi-byte> <lo-byte>      3 bytes
```

The 2-byte jump offset (16-bit) supports programs up to 65 KB of bytecode — fine for scripts. Production VMs use 4-byte offsets.
