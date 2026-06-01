#include "chunk.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>

/* Disassemble the entire chunk */
void Chunk::disassemble(const std::string &name) const {
    std::cout << "=== " << name << " ===\n";
    int offset = 0;
    while (offset < (int)code.size()) {
        offset = disassemble_instruction(offset);
    }
}

static std::string opcode_name(OpCode op) {
    switch (op) {
    case OpCode::OP_CONST:          return "OP_CONST";
    case OpCode::OP_ADD:            return "OP_ADD";
    case OpCode::OP_SUB:            return "OP_SUB";
    case OpCode::OP_MUL:            return "OP_MUL";
    case OpCode::OP_DIV:            return "OP_DIV";
    case OpCode::OP_MOD:            return "OP_MOD";
    case OpCode::OP_NEG:            return "OP_NEG";
    case OpCode::OP_NOT:            return "OP_NOT";
    case OpCode::OP_EQ:             return "OP_EQ";
    case OpCode::OP_NEQ:            return "OP_NEQ";
    case OpCode::OP_LT:             return "OP_LT";
    case OpCode::OP_GT:             return "OP_GT";
    case OpCode::OP_LE:             return "OP_LE";
    case OpCode::OP_GE:             return "OP_GE";
    case OpCode::OP_PRINT:          return "OP_PRINT";
    case OpCode::OP_POP:            return "OP_POP";
    case OpCode::OP_DEFINE_GLOBAL:  return "OP_DEFINE_GLOBAL";
    case OpCode::OP_GET_GLOBAL:     return "OP_GET_GLOBAL";
    case OpCode::OP_SET_GLOBAL:     return "OP_SET_GLOBAL";
    case OpCode::OP_JUMP:           return "OP_JUMP";
    case OpCode::OP_JUMP_IF_FALSE:  return "OP_JUMP_IF_FALSE";
    case OpCode::OP_LOOP:           return "OP_LOOP";
    case OpCode::OP_RETURN:         return "OP_RETURN";
    default:                        return "OP_UNKNOWN";
    }
}

int Chunk::disassemble_instruction(int offset) const {
    std::cout << std::setw(4) << std::setfill('0') << offset << "  ";

    /* Line number info */
    if (offset > 0 && lines[offset] == lines[offset - 1]) {
        std::cout << "   | ";
    } else {
        std::cout << std::setw(4) << lines[offset] << " ";
    }

    OpCode op = (OpCode)code[offset];
    std::string name = opcode_name(op);

    switch (op) {
    case OpCode::OP_CONST:
    case OpCode::OP_DEFINE_GLOBAL:
    case OpCode::OP_GET_GLOBAL:
    case OpCode::OP_SET_GLOBAL: {
        uint8_t idx = code[offset + 1];
        std::cout << std::left << std::setw(24) << name
                  << " " << (int)idx
                  << "  '" << value_to_string(constants[idx]) << "'\n";
        return offset + 2;
    }
    case OpCode::OP_JUMP:
    case OpCode::OP_JUMP_IF_FALSE: {
        int jump = (code[offset+1] << 8) | code[offset+2];
        std::cout << std::left << std::setw(24) << name
                  << " -> " << (offset + 3 + jump) << "\n";
        return offset + 3;
    }
    case OpCode::OP_LOOP: {
        int jump = (code[offset+1] << 8) | code[offset+2];
        std::cout << std::left << std::setw(24) << name
                  << " -> " << (offset + 3 - jump) << "\n";
        return offset + 3;
    }
    default:
        std::cout << name << "\n";
        return offset + 1;
    }
}
