#ifndef CPU_H
#define CPU_H

#include <stdint.h>
#include <stdbool.h>

/* ========================================================================
 * 16-bit CPU Architecture
 * - 8 general-purpose 16-bit registers: R0..R7
 * - Program Counter (PC), Stack Pointer (SP)
 * - Flags: ZF (zero), NF (negative), CF (carry), OF (overflow)
 * - 64KB byte-addressable memory
 * - 16-bit fixed-width instructions
 * ======================================================================== */

#define MEM_SIZE    65536   /* 64KB */
#define NUM_REGS    8
#define STACK_BASE  0xFFFE  /* stack grows downward from top of memory */

/* ---- Instruction encoding ----
 *
 *  Bits [15:12]  = opcode (4 bits → 16 opcodes; we extend with bit 11 for extras)
 *  We actually use bits [15:8] = opcode (8 bits) for a clean layout.
 *
 *  Format A  (opcode Rx, Ry)  : [15:8]=op [7:4]=Rx [3:0]=Ry
 *  Format B  (opcode Rx, imm8): [15:8]=op [7:4]=Rx [3:0]=hi(imm) + second byte
 *    — but since instructions are 16-bit, imm8 fits in the low byte directly:
 *      [15:8]=op [7:4]=Rx [3:0]=imm[7:4]  ... actually we use:
 *      [15:8]=op [7:4]=Rx [3:0]=0  and low byte = imm8  (packed differently)
 *
 *  Simplest clean encoding:
 *  Byte 0 (high):  opcode
 *  Byte 1 (low):   operands
 *
 *  For "MOV Rx, imm8": byte0=OP_MOV_IMM, byte1=[Rx(3:0) | 0]  → but imm8 needs 8 bits.
 *  Solution: treat a 16-bit instruction as:
 *    bits[15:8] = { opcode[7:1], flag }
 *    bits[7:0]  = operand byte
 *
 *  We use this layout:
 *    [15:8]  opcode (8-bit)
 *    [7:4]   reg A  (4-bit → 0..7)
 *    [3:0]   reg B or low nibble of imm
 *
 *  For IMM8 instructions a second 16-bit word holds the 16-bit address/imm.
 *  BUT since we want fixed-width 16-bit, imm8 is stored in the low byte
 *  and the register in bits [11:8].
 *
 *  Final chosen layout (simple & clean):
 *  ┌──────────┬────┬────┐
 *  │  opcode  │ RA │ RB │  (each field 4 bits; opcode 8 bits)
 *  └──────────┴────┴────┘
 *  15        8 7  4 3  0
 *
 *  Special cases:
 *  - MOV_IMM : RB field used as high nibble of imm8, second 16-bit word = full imm16
 *    → we use 32-bit instruction for MOV Rx, imm16 for simplicity.
 *    NO — to keep 16-bit fixed width, imm8 fits in RB(4 bits)+RA_low(4 bits).
 *
 *  SIMPLEST APPROACH (what we actually implement):
 *  Each "instruction" is a 16-bit value.
 *  For instructions needing an address (JMP, CALL, etc.) or imm8, we use
 *  a 16-bit word for the opcode+regs, and the NEXT 16-bit word for the immediate.
 *  The PC advances by 2 bytes always for the main word, plus 2 more if needed.
 *
 *  Instruction word:
 *    bits[15:8]  = opcode
 *    bits[7:4]   = destination register (RA)
 *    bits[3:0]   = source register (RB) OR shift amount (4-bit imm)
 *
 *  For MOV Rx, imm8  → next word (16-bit) holds the zero-extended immediate.
 *  For JMP/JEQ/.../CALL addr → next word holds the 16-bit address.
 *  For SHL/SHR Rx, imm4 → imm4 is stored in bits[3:0].
 */

typedef enum {
    /* Data movement */
    OP_MOV_IMM    = 0x01,  /* MOV Rx, imm16  (next word = imm) */
    OP_MOV_REG    = 0x02,  /* MOV Rx, Ry */
    OP_LOAD       = 0x03,  /* LOAD Rx, [Ry] */
    OP_STORE      = 0x04,  /* STORE [Rx], Ry */

    /* Arithmetic */
    OP_ADD        = 0x10,  /* ADD Rx, Ry */
    OP_SUB        = 0x11,  /* SUB Rx, Ry */
    OP_AND        = 0x12,  /* AND Rx, Ry */
    OP_OR         = 0x13,  /* OR  Rx, Ry */
    OP_XOR        = 0x14,  /* XOR Rx, Ry */
    OP_NOT        = 0x15,  /* NOT Rx (RB unused) */
    OP_SHL        = 0x16,  /* SHL Rx, imm4 (imm4 in RB field) */
    OP_SHR        = 0x17,  /* SHR Rx, imm4 */

    /* Compare */
    OP_CMP        = 0x20,  /* CMP Rx, Ry — sets flags, no store */

    /* Jumps */
    OP_JMP        = 0x30,  /* JMP addr (next word = addr) */
    OP_JEQ        = 0x31,  /* JEQ addr */
    OP_JNE        = 0x32,  /* JNE addr */
    OP_JLT        = 0x33,  /* JLT addr */
    OP_JGT        = 0x34,  /* JGT addr */

    /* Stack */
    OP_PUSH       = 0x40,  /* PUSH Rx */
    OP_POP        = 0x41,  /* POP  Rx */

    /* Call/Ret */
    OP_CALL       = 0x50,  /* CALL addr (next word = addr) */
    OP_RET        = 0x51,  /* RET */

    /* Misc */
    OP_PRINT      = 0xF0,  /* PRINT Rx — debug output */
    OP_HALT       = 0xFF,  /* HALT */
} Opcode;

/* CPU state */
typedef struct {
    uint16_t reg[NUM_REGS];  /* R0..R7 */
    uint16_t pc;             /* program counter */
    uint16_t sp;             /* stack pointer (points to next free slot) */

    /* Flags */
    bool zf;  /* zero flag */
    bool nf;  /* negative flag */
    bool cf;  /* carry flag */
    bool of;  /* overflow flag */

    bool halted;

    uint8_t mem[MEM_SIZE];   /* 64KB address space */
} CPU;

/* Function declarations */
void     cpu_init(CPU *cpu);
void     cpu_step(CPU *cpu);
void     cpu_run(CPU *cpu);
void     cpu_dump(const CPU *cpu);
void     cpu_load(CPU *cpu, const uint8_t *program, uint16_t size, uint16_t load_addr);

/* Helpers to read/write 16-bit words in memory (little-endian) */
static inline uint16_t mem_read16(const CPU *cpu, uint16_t addr) {
    return (uint16_t)(cpu->mem[addr]) | ((uint16_t)(cpu->mem[addr + 1]) << 8);
}
static inline void mem_write16(CPU *cpu, uint16_t addr, uint16_t val) {
    cpu->mem[addr]     = (uint8_t)(val & 0xFF);
    cpu->mem[addr + 1] = (uint8_t)(val >> 8);
}

#endif /* CPU_H */
