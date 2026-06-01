# CPU Emulator — Learnings

## The Fetch-Decode-Execute Cycle

Every CPU, from this toy 16-bit emulator to a modern x86-64 processor, operates on the same fundamental loop:

1. **Fetch**: Read the instruction at the address stored in the Program Counter (PC).
2. **Decode**: Interpret the bit pattern to determine what operation to perform and which registers/memory locations are involved.
3. **Execute**: Perform the operation, update registers, memory, and flags.
4. **Repeat** (increment PC to point at the next instruction first).

In `cpu_step()` in `cpu.c`, this is exactly what happens:

```c
uint16_t instr = mem_read16(cpu, cpu->pc);  // FETCH
cpu->pc += 2;
uint8_t opcode = instr >> 8;                 // DECODE
uint8_t ra     = (instr >> 4) & 0xF;
uint8_t rb     = instr & 0xF;
switch (opcode) { ... }                      // EXECUTE
```

## What the Program Counter (PC) Is

The PC is a register that holds the **address of the next instruction to execute**. After fetching an instruction, it automatically advances. For jumps and calls, we explicitly set PC to the target address — that's how control flow works.

The PC is the "instruction pointer" of the CPU. Without it, the CPU wouldn't know where in memory to look for the next instruction.

## Why Flags Matter

Flags encode information about the result of the last arithmetic or comparison operation:

- **ZF (Zero Flag)**: Set when the result is 0. Used for equality checks (`JEQ`).
- **NF (Negative Flag)**: Set when the result's sign bit is 1. Indicates negative values in two's complement.
- **CF (Carry Flag)**: Set when an unsigned addition overflows 16 bits. Used for multi-precision arithmetic.
- **OF (Overflow Flag)**: Set when a *signed* operation overflows. Example: adding two positive numbers and getting a negative result.

The key insight for conditional jumps: `CMP Rx, Ry` performs `Rx - Ry` and sets flags WITHOUT storing the result. Then `JLT` checks `NF != OF` — this is the standard x86 technique for signed comparison.

## How CALL/RET Use the Stack

A **call stack** allows functions to be called and returned from correctly, even with nested and recursive calls.

**CALL addr**:
1. Push the return address (PC+2, the instruction after the call) onto the stack: `sp -= 2; mem[sp] = pc`.
2. Jump to `addr`: `pc = addr`.

**RET**:
1. Pop the return address from the stack: `pc = mem[sp]; sp += 2`.
2. Execution continues right after the original CALL.

This creates a LIFO (Last-In-First-Out) chain: nested calls push multiple return addresses; each RET pops one, unwinding the call stack.

**Stack grows downward**: SP starts at 0xFFFE (top of memory) and decreases with each PUSH/CALL. This is standard on nearly all real architectures (x86, ARM, RISC-V).

## What an ISA Is

An **Instruction Set Architecture (ISA)** is the contract between software (programs) and hardware (the processor). It specifies:

- What instructions exist and what they do
- How instructions are encoded in binary
- How many registers there are and their width
- How memory is addressed
- The semantics of flags and condition codes

Famous ISAs include:
- **x86-64** (Intel/AMD) — complex CISC architecture, variable-length instructions
- **ARM** (phones, Apple Silicon) — RISC architecture, fixed-width 32-bit or 16-bit (Thumb) instructions
- **RISC-V** — open-source RISC ISA, clean design, used in research and increasingly production

Our toy ISA is inspired by real RISC designs: fixed-width instructions, load/store architecture (memory access only via LOAD/STORE, not in arithmetic ops), small register file.

## Two-Pass Assembly

The assembler makes two passes over the source:

1. **Pass 1** (label collection): Scan all lines without emitting code. For each label definition (`name:`), record its address. This resolves the problem of **forward references** — a `JMP loop_end` where `loop_end` is defined later in the file.

2. **Pass 2** (code generation): Emit bytecode, using the label table to resolve addresses in jump/call instructions.

This is how virtually every assembler works. GAS (GNU Assembler) and NASM both use this approach.

## Instruction Encoding Trade-offs

Our 16-bit fixed-width instruction is compact and simple to decode. The cost: immediate values are limited. For instructions needing a 16-bit address (JMP, CALL, MOV_IMM), we use an extra 16-bit word — making those instructions 32 bits wide in practice.

Real CPUs face the same trade-off:
- **x86**: variable-length instructions (1–15 bytes) — complex decoder, compact code
- **RISC-V**: 32-bit fixed-width (with optional 16-bit compressed extension)
- **THUMB/AArch32**: 16-bit and 32-bit mixed

## Memory as a Flat Array

The entire 64KB address space is one `uint8_t mem[65536]` array. This is actually how real CPUs work conceptually — the Memory Management Unit (MMU) translates virtual addresses to physical DRAM locations, but the CPU "sees" a flat address space.
