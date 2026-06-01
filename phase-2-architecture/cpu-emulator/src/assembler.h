#ifndef ASSEMBLER_H
#define ASSEMBLER_H

#include <stdint.h>

/*
 * assemble — assemble source text into bytecode
 * Returns number of bytes written, or -1 on error.
 */
int assemble(const char *source, uint8_t *out_buf, int buf_size, uint16_t base_addr);

/*
 * assemble_file — read a .asm file and assemble it
 */
int assemble_file(const char *filename, uint8_t *out_buf, int buf_size, uint16_t base_addr);

#endif /* ASSEMBLER_H */
