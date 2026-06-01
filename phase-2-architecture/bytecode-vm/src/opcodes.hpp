#pragma once
#include <cstdint>

/*
 * opcodes.hpp — Opcode definitions for the Lox-mini bytecode VM
 *
 * This is a stack-based VM (like CPython, the JVM in interpreted mode,
 * and the original Lua VM). Stack machines are simpler than register
 * machines because:
 *   - No register allocation needed during compilation
 *   - Instructions are implicit: operands are always on the stack
 *   - Compact bytecode (no need to specify source/dest registers)
 *
 * Each opcode is 1 byte. Some opcodes take 1-byte operands.
 */
enum class OpCode : uint8_t {
    // Push constant from pool: OP_CONST <index>
    OP_CONST,

    // Arithmetic (pop 2, push 1)
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_MOD,

    // Unary (pop 1, push 1)
    OP_NEG,
    OP_NOT,

    // Comparison (pop 2, push bool)
    OP_EQ,
    OP_NEQ,
    OP_LT,
    OP_GT,
    OP_LE,
    OP_GE,

    // Output
    OP_PRINT,

    // Stack manipulation
    OP_POP,

    // Variables: <name_index> — index into constant pool for the name string
    OP_DEFINE_GLOBAL,   // pop value, define global variable
    OP_GET_GLOBAL,      // push value of global
    OP_SET_GLOBAL,      // pop value, set global (must already exist)

    // Control flow
    // OP_JUMP <offset_hi> <offset_lo>  — unconditional jump forward by offset bytes
    OP_JUMP,
    // OP_JUMP_IF_FALSE <offset_hi> <offset_lo> — pop bool, jump if false
    OP_JUMP_IF_FALSE,
    // OP_LOOP <offset_hi> <offset_lo> — jump backward by offset bytes
    OP_LOOP,

    // Function return
    OP_RETURN,
};
