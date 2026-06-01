#include "cpu.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

/* ========================================================================
 * cpu_init — zero-out the CPU state, set SP to top of memory
 * ======================================================================== */
void cpu_init(CPU *cpu) {
    memset(cpu, 0, sizeof(*cpu));
    cpu->sp     = STACK_BASE;
    cpu->pc     = 0;
    cpu->halted = false;
}

/* ========================================================================
 * cpu_load — copy a program into memory at load_addr
 * ======================================================================== */
void cpu_load(CPU *cpu, const uint8_t *program, uint16_t size, uint16_t load_addr) {
    if ((uint32_t)load_addr + size > MEM_SIZE) {
        fprintf(stderr, "Program too large for memory.\n");
        exit(1);
    }
    memcpy(cpu->mem + load_addr, program, size);
    cpu->pc = load_addr;
}

/* ========================================================================
 * update_flags — set ZF, NF after an arithmetic result
 *   result32 : the 32-bit result (to detect carry out of bit 15)
 *   a, b     : original operands (16-bit signed, for overflow detection)
 *   is_sub   : true if this was a subtraction
 * ======================================================================== */
static void update_flags(CPU *cpu, uint32_t result32, int32_t a, int32_t b, bool is_sub) {
    uint16_t result16 = (uint16_t)(result32 & 0xFFFF);

    cpu->zf = (result16 == 0);
    cpu->nf = (result16 & 0x8000) != 0;      /* bit 15 = sign bit */
    cpu->cf = (result32 > 0xFFFF);            /* unsigned carry */

    /* Signed overflow: sign of both operands same, but result differs */
    int32_t b_eff = is_sub ? -b : b;          /* for subtraction, b is negated */
    bool a_neg    = (a & 0x8000) != 0;
    bool b_neg    = (b_eff & 0x8000) != 0;
    bool r_neg    = (result16 & 0x8000) != 0;
    cpu->of = (a_neg == b_neg) && (r_neg != a_neg);
}

/* ========================================================================
 * cpu_step — fetch, decode, and execute one instruction
 * ======================================================================== */
void cpu_step(CPU *cpu) {
    if (cpu->halted) return;

    /* Fetch the 16-bit instruction word */
    uint16_t instr = mem_read16(cpu, cpu->pc);
    cpu->pc += 2;

    uint8_t  opcode = (uint8_t)(instr >> 8);
    uint8_t  ra     = (uint8_t)((instr >> 4) & 0x0F);
    uint8_t  rb     = (uint8_t)(instr & 0x0F);

    /* Validate register indices */
    if (ra >= NUM_REGS || rb >= NUM_REGS) {
        fprintf(stderr, "Invalid register index at PC=0x%04X\n", cpu->pc - 2);
        cpu->halted = true;
        return;
    }

    uint16_t imm;   /* immediate / address operand from next word */
    uint32_t res32;

    switch ((Opcode)opcode) {

    /* ---- Data movement ---- */
    case OP_MOV_IMM:
        imm = mem_read16(cpu, cpu->pc);
        cpu->pc += 2;
        cpu->reg[ra] = imm;
        break;

    case OP_MOV_REG:
        cpu->reg[ra] = cpu->reg[rb];
        break;

    case OP_LOAD:
        cpu->reg[ra] = mem_read16(cpu, cpu->reg[rb]);
        break;

    case OP_STORE:
        mem_write16(cpu, cpu->reg[ra], cpu->reg[rb]);
        break;

    /* ---- Arithmetic ---- */
    case OP_ADD:
        res32 = (uint32_t)cpu->reg[ra] + (uint32_t)cpu->reg[rb];
        update_flags(cpu, res32, (int32_t)cpu->reg[ra], (int32_t)cpu->reg[rb], false);
        cpu->reg[ra] = (uint16_t)(res32 & 0xFFFF);
        break;

    case OP_SUB:
        res32 = (uint32_t)cpu->reg[ra] - (uint32_t)cpu->reg[rb];
        update_flags(cpu, res32, (int32_t)cpu->reg[ra], (int32_t)cpu->reg[rb], true);
        cpu->reg[ra] = (uint16_t)(res32 & 0xFFFF);
        break;

    case OP_AND:
        cpu->reg[ra] &= cpu->reg[rb];
        cpu->zf = (cpu->reg[ra] == 0);
        cpu->nf = (cpu->reg[ra] & 0x8000) != 0;
        cpu->cf = cpu->of = false;
        break;

    case OP_OR:
        cpu->reg[ra] |= cpu->reg[rb];
        cpu->zf = (cpu->reg[ra] == 0);
        cpu->nf = (cpu->reg[ra] & 0x8000) != 0;
        cpu->cf = cpu->of = false;
        break;

    case OP_XOR:
        cpu->reg[ra] ^= cpu->reg[rb];
        cpu->zf = (cpu->reg[ra] == 0);
        cpu->nf = (cpu->reg[ra] & 0x8000) != 0;
        cpu->cf = cpu->of = false;
        break;

    case OP_NOT:
        cpu->reg[ra] = ~cpu->reg[ra];
        cpu->zf = (cpu->reg[ra] == 0);
        cpu->nf = (cpu->reg[ra] & 0x8000) != 0;
        cpu->cf = cpu->of = false;
        break;

    case OP_SHL: {
        uint8_t shift = rb & 0x0F;
        uint32_t shifted = (uint32_t)cpu->reg[ra] << shift;
        cpu->cf = (shifted > 0xFFFF);
        cpu->reg[ra] = (uint16_t)(shifted & 0xFFFF);
        cpu->zf = (cpu->reg[ra] == 0);
        cpu->nf = (cpu->reg[ra] & 0x8000) != 0;
        break;
    }

    case OP_SHR: {
        uint8_t shift = rb & 0x0F;
        cpu->cf = shift > 0 ? ((cpu->reg[ra] >> (shift - 1)) & 1) : 0;
        cpu->reg[ra] >>= shift;
        cpu->zf = (cpu->reg[ra] == 0);
        cpu->nf = (cpu->reg[ra] & 0x8000) != 0;
        break;
    }

    /* ---- Compare ---- */
    case OP_CMP: {
        res32 = (uint32_t)cpu->reg[ra] - (uint32_t)cpu->reg[rb];
        update_flags(cpu, res32, (int32_t)cpu->reg[ra], (int32_t)cpu->reg[rb], true);
        /* do NOT store result */
        break;
    }

    /* ---- Jumps ---- */
    case OP_JMP:
        imm = mem_read16(cpu, cpu->pc);
        cpu->pc = imm;  /* jump to address directly (not +2) */
        break;

    case OP_JEQ:
        imm = mem_read16(cpu, cpu->pc);
        cpu->pc = cpu->zf ? imm : (uint16_t)(cpu->pc + 2);
        break;

    case OP_JNE:
        imm = mem_read16(cpu, cpu->pc);
        cpu->pc = !cpu->zf ? imm : (uint16_t)(cpu->pc + 2);
        break;

    case OP_JLT:
        /* Signed less-than: NF != OF (or just NF after CMP, since we do Rx-Ry) */
        imm = mem_read16(cpu, cpu->pc);
        cpu->pc = (cpu->nf != cpu->of) ? imm : (uint16_t)(cpu->pc + 2);
        break;

    case OP_JGT:
        /* Signed greater-than: not ZF and (NF == OF) */
        imm = mem_read16(cpu, cpu->pc);
        cpu->pc = (!cpu->zf && (cpu->nf == cpu->of)) ? imm : (uint16_t)(cpu->pc + 2);
        break;

    /* ---- Stack ---- */
    case OP_PUSH:
        cpu->sp -= 2;
        mem_write16(cpu, cpu->sp, cpu->reg[ra]);
        break;

    case OP_POP:
        cpu->reg[ra] = mem_read16(cpu, cpu->sp);
        cpu->sp += 2;
        break;

    /* ---- Call / Return ---- */
    case OP_CALL:
        imm = mem_read16(cpu, cpu->pc);
        cpu->pc += 2;           /* PC now points past the address word */
        cpu->sp -= 2;
        mem_write16(cpu, cpu->sp, cpu->pc);  /* push return address */
        cpu->pc = imm;          /* jump to function */
        break;

    case OP_RET:
        cpu->pc = mem_read16(cpu, cpu->sp);  /* pop return address */
        cpu->sp += 2;
        break;

    /* ---- Misc ---- */
    case OP_PRINT:
        printf("R%d = %u (0x%04X, signed: %d)\n",
               ra,
               (unsigned)cpu->reg[ra],
               (unsigned)cpu->reg[ra],
               (int16_t)cpu->reg[ra]);
        break;

    case OP_HALT:
        cpu->halted = true;
        break;

    default:
        fprintf(stderr, "Unknown opcode 0x%02X at PC=0x%04X\n", opcode, cpu->pc - 2);
        cpu->halted = true;
        break;
    }
}

/* ========================================================================
 * cpu_run — run until HALT or error (max 10M instructions as safety)
 * ======================================================================== */
void cpu_run(CPU *cpu) {
    int limit = 10000000;
    while (!cpu->halted && limit-- > 0) {
        cpu_step(cpu);
    }
    if (limit <= 0) {
        fprintf(stderr, "Execution limit reached — possible infinite loop.\n");
    }
}

/* ========================================================================
 * cpu_dump — print all CPU registers and flags
 * ======================================================================== */
void cpu_dump(const CPU *cpu) {
    printf("\n=== CPU STATE DUMP ===\n");
    for (int i = 0; i < NUM_REGS; i++) {
        printf("  R%d = 0x%04X (%5u, signed: %6d)\n",
               i,
               (unsigned)cpu->reg[i],
               (unsigned)cpu->reg[i],
               (int16_t)cpu->reg[i]);
    }
    printf("  PC = 0x%04X\n", (unsigned)cpu->pc);
    printf("  SP = 0x%04X\n", (unsigned)cpu->sp);
    printf("  Flags: ZF=%d NF=%d CF=%d OF=%d\n",
           cpu->zf, cpu->nf, cpu->cf, cpu->of);
    printf("  Halted: %s\n", cpu->halted ? "yes" : "no");
    printf("======================\n\n");
}
