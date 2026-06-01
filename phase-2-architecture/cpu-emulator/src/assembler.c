/*
 * assembler.c — Minimal two-pass assembler for our 16-bit CPU
 *
 * Pass 1: scan for labels, build a symbol table with their addresses.
 * Pass 2: emit bytecode, resolving label references.
 *
 * Assembly syntax:
 *   label:                     ; defines a label at current address
 *   MOV R0, 42                 ; MOV Rx, imm
 *   MOV R0, R1                 ; MOV Rx, Ry
 *   LOAD R0, [R1]              ; LOAD Rx, [Ry]
 *   STORE [R0], R1             ; STORE [Rx], Ry
 *   ADD R0, R1
 *   SUB R0, R1
 *   AND R0, R1 / OR R0, R1 / XOR R0, R1 / NOT R0
 *   SHL R0, 4 / SHR R0, 2
 *   CMP R0, R1
 *   JMP label_or_addr
 *   JEQ/JNE/JLT/JGT label_or_addr
 *   PUSH R0 / POP R0
 *   CALL label_or_addr / RET
 *   PRINT R0
 *   HALT
 *   ; comment
 */

#include "cpu.h"
#include "assembler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>

#define MAX_LABELS  256
#define MAX_LINE    256
#define MAX_OUTPUT  65536

typedef struct {
    char     name[64];
    uint16_t addr;
} Label;

typedef struct {
    Label    labels[MAX_LABELS];
    int      label_count;

    uint8_t  output[MAX_OUTPUT];
    int      out_pos;

    uint16_t base_addr;  /* where in memory the program is loaded */
} Assembler;

/* ---- Utilities ---- */

static void asm_init(Assembler *a, uint16_t base_addr) {
    memset(a, 0, sizeof(*a));
    a->base_addr = base_addr;
}

static void trim(char *s) {
    /* Remove leading whitespace */
    char *p = s;
    while (*p && isspace((unsigned char)*p)) p++;
    if (p != s) memmove(s, p, strlen(p) + 1);
    /* Remove trailing whitespace and newline */
    int len = (int)strlen(s);
    while (len > 0 && isspace((unsigned char)s[len-1])) s[--len] = '\0';
}

static void strip_comment(char *s) {
    char *p = strchr(s, ';');
    if (p) *p = '\0';
    trim(s);
}

/* Parse "Rn" → register number, or -1 on error */
static int parse_reg(const char *s) {
    if ((s[0] == 'R' || s[0] == 'r') && s[1] >= '0' && s[1] <= '7' && s[2] == '\0')
        return s[1] - '0';
    return -1;
}

/* Parse integer literal (decimal or 0x hex) */
static uint16_t parse_imm(const char *s) {
    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
        return (uint16_t)strtol(s, NULL, 16);
    return (uint16_t)atoi(s);
}

/* Look up label address; return 0xFFFF if not found */
static uint16_t find_label(Assembler *a, const char *name) {
    for (int i = 0; i < a->label_count; i++) {
        if (strcmp(a->labels[i].name, name) == 0)
            return a->labels[i].addr;
    }
    return 0xFFFF;
}

static void add_label(Assembler *a, const char *name, uint16_t addr) {
    if (a->label_count >= MAX_LABELS) {
        fprintf(stderr, "Too many labels\n");
        exit(1);
    }
    strncpy(a->labels[a->label_count].name, name, 63);
    a->labels[a->label_count].addr = addr;
    a->label_count++;
}

/* Emit one byte */
static void emit8(Assembler *a, uint8_t b) {
    if (a->out_pos >= MAX_OUTPUT) { fprintf(stderr, "Output overflow\n"); exit(1); }
    a->output[a->out_pos++] = b;
}

/* Emit a 16-bit word (little-endian) */
static void emit16(Assembler *a, uint16_t w) {
    emit8(a, (uint8_t)(w & 0xFF));
    emit8(a, (uint8_t)(w >> 8));
}

/* Emit an instruction word: [opcode | (ra<<4) | rb] as little-endian 16-bit */
static void emit_instr(Assembler *a, uint8_t opcode, uint8_t ra, uint8_t rb) {
    uint16_t word = ((uint16_t)opcode << 8) | ((uint16_t)(ra & 0xF) << 4) | (rb & 0xF);
    emit16(a, word);
}

/* ========================================================================
 * split_operands — split "R0, R1" into two tokens
 * ======================================================================== */
static void split_operands(char *ops, char *tok1, char *tok2) {
    tok1[0] = tok2[0] = '\0';
    char *comma = strchr(ops, ',');
    if (comma) {
        *comma = '\0';
        strncpy(tok1, ops,    63); trim(tok1);
        strncpy(tok2, comma+1, 63); trim(tok2);
    } else {
        strncpy(tok1, ops, 63); trim(tok1);
    }
}

/* ========================================================================
 * assemble_line — emit bytecode for one line (used in both passes)
 *   pass == 1: just count bytes (don't emit)
 *   pass == 2: emit bytes
 * ======================================================================== */
static void assemble_line(Assembler *a, int pass, const char *mnemonic, const char *ops_in) {
    char ops[MAX_LINE];
    strncpy(ops, ops_in, MAX_LINE-1);
    trim(ops);

    char tok1[64], tok2[64];
    split_operands(ops, tok1, tok2);

    int ra = tok1[0] ? parse_reg(tok1) : 0;
    int rb = tok2[0] ? parse_reg(tok2) : 0;

    /* Helper: emit or skip */
#define DO_EMIT_INSTR(op, a_, b_) do { \
    if (pass == 2) emit_instr(a, (op), (a_), (b_)); \
    else a->out_pos += 2; \
} while(0)

#define DO_EMIT16(w) do { \
    if (pass == 2) emit16(a, (w)); \
    else a->out_pos += 2; \
} while(0)

    if (strcmp(mnemonic, "MOV") == 0) {
        if (rb >= 0) {
            /* MOV Rx, Ry */
            DO_EMIT_INSTR(OP_MOV_REG, ra, rb);
        } else {
            /* MOV Rx, imm — tok2 is an immediate value */
            uint16_t imm = parse_imm(tok2);
            DO_EMIT_INSTR(OP_MOV_IMM, ra, 0);
            DO_EMIT16(imm);
        }
    } else if (strcmp(mnemonic, "LOAD") == 0) {
        /* LOAD Rx, [Ry] — tok2 looks like "[R1]" */
        char regname[8] = {0};
        sscanf(tok2, "[%7[^]]]", regname);
        rb = parse_reg(regname);
        DO_EMIT_INSTR(OP_LOAD, ra, rb);
    } else if (strcmp(mnemonic, "STORE") == 0) {
        /* STORE [Rx], Ry — tok1 looks like "[R0]" */
        char regname[8] = {0};
        sscanf(tok1, "[%7[^]]]", regname);
        ra = parse_reg(regname);
        rb = parse_reg(tok2);
        DO_EMIT_INSTR(OP_STORE, ra, rb);
    } else if (strcmp(mnemonic, "ADD") == 0) {
        DO_EMIT_INSTR(OP_ADD, ra, rb);
    } else if (strcmp(mnemonic, "SUB") == 0) {
        DO_EMIT_INSTR(OP_SUB, ra, rb);
    } else if (strcmp(mnemonic, "AND") == 0) {
        DO_EMIT_INSTR(OP_AND, ra, rb);
    } else if (strcmp(mnemonic, "OR") == 0) {
        DO_EMIT_INSTR(OP_OR, ra, rb);
    } else if (strcmp(mnemonic, "XOR") == 0) {
        DO_EMIT_INSTR(OP_XOR, ra, rb);
    } else if (strcmp(mnemonic, "NOT") == 0) {
        DO_EMIT_INSTR(OP_NOT, ra, 0);
    } else if (strcmp(mnemonic, "SHL") == 0) {
        uint8_t imm4 = (uint8_t)(atoi(tok2) & 0xF);
        DO_EMIT_INSTR(OP_SHL, ra, imm4);
    } else if (strcmp(mnemonic, "SHR") == 0) {
        uint8_t imm4 = (uint8_t)(atoi(tok2) & 0xF);
        DO_EMIT_INSTR(OP_SHR, ra, imm4);
    } else if (strcmp(mnemonic, "CMP") == 0) {
        DO_EMIT_INSTR(OP_CMP, ra, rb);
    } else if (strcmp(mnemonic, "JMP") == 0) {
        DO_EMIT_INSTR(OP_JMP, 0, 0);
        uint16_t addr = (pass == 2) ? (
            (tok1[0] >= '0' && tok1[0] <= '9') ? parse_imm(tok1) : find_label(a, tok1)
        ) : 0;
        DO_EMIT16(addr);
    } else if (strcmp(mnemonic, "JEQ") == 0) {
        DO_EMIT_INSTR(OP_JEQ, 0, 0);
        uint16_t addr = (pass == 2) ? (
            (tok1[0] >= '0' && tok1[0] <= '9') ? parse_imm(tok1) : find_label(a, tok1)
        ) : 0;
        DO_EMIT16(addr);
    } else if (strcmp(mnemonic, "JNE") == 0) {
        DO_EMIT_INSTR(OP_JNE, 0, 0);
        uint16_t addr = (pass == 2) ? (
            (tok1[0] >= '0' && tok1[0] <= '9') ? parse_imm(tok1) : find_label(a, tok1)
        ) : 0;
        DO_EMIT16(addr);
    } else if (strcmp(mnemonic, "JLT") == 0) {
        DO_EMIT_INSTR(OP_JLT, 0, 0);
        uint16_t addr = (pass == 2) ? (
            (tok1[0] >= '0' && tok1[0] <= '9') ? parse_imm(tok1) : find_label(a, tok1)
        ) : 0;
        DO_EMIT16(addr);
    } else if (strcmp(mnemonic, "JGT") == 0) {
        DO_EMIT_INSTR(OP_JGT, 0, 0);
        uint16_t addr = (pass == 2) ? (
            (tok1[0] >= '0' && tok1[0] <= '9') ? parse_imm(tok1) : find_label(a, tok1)
        ) : 0;
        DO_EMIT16(addr);
    } else if (strcmp(mnemonic, "PUSH") == 0) {
        DO_EMIT_INSTR(OP_PUSH, ra, 0);
    } else if (strcmp(mnemonic, "POP") == 0) {
        DO_EMIT_INSTR(OP_POP, ra, 0);
    } else if (strcmp(mnemonic, "CALL") == 0) {
        DO_EMIT_INSTR(OP_CALL, 0, 0);
        uint16_t addr = (pass == 2) ? (
            (tok1[0] >= '0' && tok1[0] <= '9') ? parse_imm(tok1) : find_label(a, tok1)
        ) : 0;
        DO_EMIT16(addr);
    } else if (strcmp(mnemonic, "RET") == 0) {
        DO_EMIT_INSTR(OP_RET, 0, 0);
    } else if (strcmp(mnemonic, "PRINT") == 0) {
        DO_EMIT_INSTR(OP_PRINT, ra, 0);
    } else if (strcmp(mnemonic, "HALT") == 0) {
        DO_EMIT_INSTR(OP_HALT, 0, 0);
    } else {
        fprintf(stderr, "Unknown mnemonic: %s\n", mnemonic);
        exit(1);
    }

#undef DO_EMIT_INSTR
#undef DO_EMIT16
}

/* ========================================================================
 * assemble — two-pass assembly from source text to bytecode
 *   Returns number of bytes written, or -1 on error.
 * ======================================================================== */
int assemble(const char *source, uint8_t *out_buf, int buf_size, uint16_t base_addr) {
    Assembler a;
    asm_init(&a, base_addr);

    /* ---- PASS 1: collect labels ---- */
    {
        char *src_copy = strdup(source);
        char *line = strtok(src_copy, "\n");
        while (line) {
            char buf[MAX_LINE];
            strncpy(buf, line, MAX_LINE-1);
            strip_comment(buf);
            trim(buf);

            if (buf[0] == '\0') { line = strtok(NULL, "\n"); continue; }

            /* Check for label definition (ends with ':') */
            int len = (int)strlen(buf);
            if (buf[len-1] == ':') {
                buf[len-1] = '\0';
                trim(buf);
                add_label(&a, buf, (uint16_t)(base_addr + a.out_pos));
                line = strtok(NULL, "\n");
                continue;
            }

            /* Parse mnemonic and operands */
            char mnemonic[32];
            char rest[MAX_LINE] = "";
            sscanf(buf, "%31s%*[ \t]%[^\n]", mnemonic, rest);
            trim(rest);

            /* Count bytes for this instruction (pass=1) */
            assemble_line(&a, 1, mnemonic, rest);

            line = strtok(NULL, "\n");
        }
        free(src_copy);
    }

    /* ---- PASS 2: emit bytecode ---- */
    a.out_pos = 0;
    {
        char *src_copy = strdup(source);
        char *line = strtok(src_copy, "\n");
        while (line) {
            char buf[MAX_LINE];
            strncpy(buf, line, MAX_LINE-1);
            strip_comment(buf);
            trim(buf);

            if (buf[0] == '\0') { line = strtok(NULL, "\n"); continue; }

            int len = (int)strlen(buf);
            if (buf[len-1] == ':') {  /* label — skip */
                line = strtok(NULL, "\n");
                continue;
            }

            char mnemonic[32];
            char rest[MAX_LINE] = "";
            sscanf(buf, "%31s%*[ \t]%[^\n]", mnemonic, rest);
            trim(rest);

            assemble_line(&a, 2, mnemonic, rest);

            line = strtok(NULL, "\n");
        }
        free(src_copy);
    }

    if (a.out_pos > buf_size) {
        fprintf(stderr, "Output buffer too small\n");
        return -1;
    }
    memcpy(out_buf, a.output, a.out_pos);
    return a.out_pos;
}

/* ========================================================================
 * assemble_file — read a .asm file and assemble it
 * ======================================================================== */
int assemble_file(const char *filename, uint8_t *out_buf, int buf_size, uint16_t base_addr) {
    FILE *f = fopen(filename, "r");
    if (!f) { perror(filename); return -1; }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);

    char *src = malloc(fsize + 1);
    if (!src) { fclose(f); return -1; }
    fread(src, 1, fsize, f);
    src[fsize] = '\0';
    fclose(f);

    int result = assemble(src, out_buf, buf_size, base_addr);
    free(src);
    return result;
}
