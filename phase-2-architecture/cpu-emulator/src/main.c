#include "cpu.h"
#include "assembler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s <program.asm>\n", prog);
    fprintf(stderr, "       %s --demo\n", prog);
    fprintf(stderr, "\nExamples:\n");
    fprintf(stderr, "  %s src/programs/fibonacci.asm\n", prog);
    fprintf(stderr, "  %s src/programs/factorial.asm\n", prog);
    fprintf(stderr, "  %s src/programs/bubble_sort.asm\n", prog);
}

/* Run a quick inline demo to verify the CPU works */
static void run_demo(void) {
    printf("=== CPU Emulator Demo ===\n\n");

    /* Assemble a small program inline */
    const char *demo_src =
        "; Demo: compute 3 + 5 and print result\n"
        "    MOV R0, 3\n"
        "    MOV R1, 5\n"
        "    ADD R0, R1\n"     /* R0 = 8 */
        "    PRINT R0\n"
        "; Demo: show flags\n"
        "    MOV R2, 8\n"
        "    CMP R0, R2\n"     /* ZF should be set */
        "; Demo: loop counting down from 5\n"
        "    MOV R3, 5\n"
        "    MOV R4, 0\n"
        "loop:\n"
        "    CMP R3, R4\n"
        "    JEQ done\n"
        "    PRINT R3\n"
        "    MOV R5, 1\n"
        "    SUB R3, R5\n"
        "    JMP loop\n"
        "done:\n"
        "    HALT\n";

    uint8_t bytecode[4096];
    int size = assemble(demo_src, bytecode, sizeof(bytecode), 0x0000);
    if (size < 0) {
        fprintf(stderr, "Assembly failed.\n");
        return;
    }
    printf("Assembled %d bytes.\n\n", size);

    CPU cpu;
    cpu_init(&cpu);
    cpu_load(&cpu, bytecode, (uint16_t)size, 0x0000);
    cpu_run(&cpu);
    cpu_dump(&cpu);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "--demo") == 0) {
        run_demo();
        return 0;
    }

    const char *filename = argv[1];
    uint8_t bytecode[65536];
    int size = assemble_file(filename, bytecode, sizeof(bytecode), 0x0000);
    if (size < 0) {
        fprintf(stderr, "Failed to assemble '%s'\n", filename);
        return 1;
    }

    printf("Assembled '%s': %d bytes\n\n", filename, size);

    CPU cpu;
    cpu_init(&cpu);
    cpu_load(&cpu, bytecode, (uint16_t)size, 0x0000);

    printf("--- Output ---\n");
    cpu_run(&cpu);
    printf("--------------\n");
    cpu_dump(&cpu);

    return 0;
}
