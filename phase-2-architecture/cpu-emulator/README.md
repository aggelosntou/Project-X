# CPU Emulator

A complete 16-bit CPU emulator written in C, with an assembler and three demo programs.

## Architecture

- 8 general-purpose 16-bit registers: R0–R7
- Special registers: PC (program counter), SP (stack pointer)
- Flags: ZF (zero), NF (negative), CF (carry), OF (overflow)
- 64KB byte-addressable memory
- 16-bit fixed-width instruction encoding

## Instruction Set

| Instruction | Description |
|---|---|
| `MOV Rx, imm` | Load immediate value into register |
| `MOV Rx, Ry` | Copy register to register |
| `LOAD Rx, [Ry]` | Load 16-bit word from memory address in Ry |
| `STORE [Rx], Ry` | Store Ry to memory address in Rx |
| `ADD Rx, Ry` | Rx = Rx + Ry, set flags |
| `SUB Rx, Ry` | Rx = Rx - Ry, set flags |
| `AND/OR/XOR Rx, Ry` | Bitwise operations |
| `NOT Rx` | Bitwise NOT |
| `SHL/SHR Rx, imm4` | Bit shifts |
| `CMP Rx, Ry` | Set flags based on Rx - Ry (no store) |
| `JMP addr` | Unconditional jump |
| `JEQ/JNE/JLT/JGT addr` | Conditional jumps |
| `PUSH Rx / POP Rx` | Stack operations |
| `CALL addr / RET` | Subroutine call and return |
| `PRINT Rx` | Debug: print register value |
| `HALT` | Stop execution |

## Building

```bash
make
```

## Running Programs

```bash
make run-demo          # Quick inline demo
make run-fibonacci     # First 10 Fibonacci numbers
make run-factorial     # Factorial computation
make run-bubblesort    # Bubble sort of 8 numbers
```

Or directly:

```bash
./cpu_emulator src/programs/fibonacci.asm
```

## Instruction Encoding

Each instruction is a 16-bit word:

```
┌──────────┬────┬────┐
│  opcode  │ RA │ RB │
└──────────┴────┴────┘
 15      8  7  4 3  0
```

Instructions that need an address or immediate use the following 16-bit word as the operand.

## Assembly Syntax

```asm
; This is a comment
label:
    MOV R0, 42          ; immediate
    MOV R1, R0          ; register copy
    ADD R0, R1          ; R0 = R0 + R1
    CMP R0, R1          ; set flags
    JEQ done            ; jump if equal
    PUSH R0             ; push to stack
    CALL my_func        ; call subroutine
    HALT
my_func:
    RET
```
