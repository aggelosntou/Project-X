# Compiler Architecture — Key Concepts

## 1. Why Separate Lexer / Parser / Codegen Passes?

Each pass transforms the program from one representation to a cleaner,
more structured one:

```
Characters → (lexer) → Tokens → (parser) → AST → (codegen) → Assembly
```

**Separation of concerns:**
- The lexer only cares about individual characters and token shapes.
  It does not know whether `x = y + z` is syntactically valid.
- The parser only cares about grammar structure.
  It does not know what registers exist.
- The codegen only cares about emitting correct instructions.
  It has already been handed a fully-validated, well-structured AST.

**Practical benefits:**
- You can swap the backend (x86 → ARM → WASM) by replacing only codegen.c.
- You can add a type-checker, optimizer, or interpreter between parser and
  codegen without touching the lexer.
- Errors are reported at the right level: "unexpected character '@'" (lexer)
  vs "expected ';'" (parser) vs "undefined variable 'foo'" (codegen/typecheck).

---

## 2. What Is an AST?

An **Abstract Syntax Tree** is a tree where:
- Each **node** represents one grammatical construct.
- **Children** are the sub-constructs.
- Concrete syntax details (parentheses, semicolons, whitespace) are thrown away.

Example: `x + y * 2` becomes:

```
BINOP "+"
  IDENT "x"
  BINOP "*"
    IDENT "y"
    NUMBER 2
```

The tree structure encodes operator precedence: `*` is deeper than `+`,
so it will be evaluated first. No parentheses needed in the AST.

### Why does the AST decouple parsing from code generation?

Once the parser hands over an AST, the codegen no longer needs to re-parse
or re-determine precedence. It simply recursively walks the tree.
This is much easier to test: you can print the AST (`CALC_DUMP_AST=1`)
and inspect it without running any code generation.

---

## 3. Recursive-Descent Parsing

Each grammar rule becomes one C function:

```
expr     → or_expr
or_expr  → and_expr  ( "||" and_expr  )*
and_expr → eq_expr   ( "&&" eq_expr   )*
...
primary  → NUMBER | IDENT | "(" expr ")"
```

**Left recursion elimination**: naive grammars write `expr → expr + term`
which causes infinite recursion. We convert to iterative style:
`add_expr` calls `mul_expr` and then loops on `+` / `-` tokens.

**Precedence** is determined by the call stack depth:
- `or_expr` calls `and_expr` — so `&&` binds tighter than `||`
- `mul_expr` calls `unary`   — so `*` binds tighter than `+`
- `primary` is at the bottom — atoms bind tightest

---

## 4. System V AMD64 ABI

The **System V AMD64 ABI** (Application Binary Interface) defines the
calling convention on Linux and macOS for x86-64 code:

| Register | Role                                                  |
|----------|-------------------------------------------------------|
| %rdi     | 1st function argument                                 |
| %rsi     | 2nd function argument                                 |
| %rdx     | 3rd argument / high half of dividend for idiv         |
| %rcx     | 4th argument                                          |
| %r8      | 5th argument                                          |
| %r9      | 6th argument                                          |
| %rax     | Return value / accumulator                            |
| %rbp     | Frame base pointer (callee-saved)                     |
| %rsp     | Stack pointer (callee-saved by convention)            |
| %rbx, %r12–%r15 | Callee-saved (must be preserved across calls) |

**Caller-saved** registers (%rax, %rcx, %rdx, %rsi, %rdi, %r8–%r11):
  the callee may trash them; the caller must save them if needed.

**Callee-saved** registers (%rbp, %rbx, %r12–%r15):
  the callee must restore them before returning.

For `printf("%ld\n", value)`:
- %rdi = address of format string
- %rsi = value to print
- %eax = 0 (signals "no XMM vector arguments used")

---

## 5. Why %rax Is the Accumulator

`%rax` is the "accumulator" register by historical convention dating back to
the 8086. In x86-64:
- `ret` returns the value in `%rax` to the caller.
- `idivq` divides `%rdx:%rax` by its operand, result in `%rax`.
- `imulq` can use `%rax` as an implicit operand.

Our codegen uses %rax as the single "result register": every expression
leaves its value in %rax. To evaluate `left + right`:
1. Evaluate left → %rax
2. Push %rax (save left on stack)
3. Evaluate right → %rax
4. Pop %rcx (restore left)
5. `addq %rcx, %rax` → result in %rax

This is the simplest possible code generation strategy. Real compilers use
**register allocation** to keep intermediate values in registers instead of
pushing/popping, avoiding memory traffic.

---

## 6. Division on x86-64

`idivq %rdi` divides the 128-bit value `%rdx:%rax` by `%rdi`:
- Quotient  → %rax
- Remainder → %rdx

Before calling `idivq`, you must:
1. Set the dividend in %rax.
2. Sign-extend it into %rdx with `cqto` (formerly `cqo`/`cdq`).

Our codegen for `a / b`:
```asm
# After emit_expr(left) → %rax, then pushq %rax
# After emit_expr(right) → %rax, then popq %rcx
#   %rcx = left (dividend), %rax = right (divisor)
movq  %rax, %rdi     # move divisor out of %rax (needed by idivq)
movq  %rcx, %rax     # dividend into %rax
cqto                 # sign-extend rax → rdx:rax
idivq %rdi           # quotient in %rax
```

---

## 7. Comparison and Boolean Values

In Calc, comparisons produce integer 0 (false) or 1 (true).
x86-64 `cmpq` sets CPU flags without storing a result. The `setX` family
stores the flag result into a byte register:

```asm
cmpq  %rax, %rcx      # computes %rcx - %rax and sets flags
setl  %al             # AL = 1 if %rcx < %rax (signed less)
movzbq %al, %rax      # zero-extend AL to 64-bit %rax
```

The comparison order matters: `cmpq src, dst` computes `dst - src`.
So `cmpq %rax, %rcx` tests `%rcx op %rax`, i.e. `left op right`. ✓

---

## 8. Stack Frame Layout

```
Higher addresses
+------------------+
| ... caller frame |
+------------------+  ← old %rsp (before call)
| return address   |  pushed by `call main`
+------------------+  ← %rsp after `call`
| saved %rbp       |  pushed by `pushq %rbp`
+------------------+  ← %rbp  (after `movq %rsp, %rbp`)
| var x  (-8 %rbp) |  8 bytes
| var y (-16 %rbp) |  8 bytes
| ...              |
+------------------+  ← %rsp  (after `subq $N, %rsp`)
Lower addresses
```

The `leave` instruction is equivalent to:
```asm
movq  %rbp, %rsp    # restore stack pointer
popq  %rbp          # restore caller's frame pointer
```
