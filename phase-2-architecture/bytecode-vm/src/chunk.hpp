#pragma once
#include "opcodes.hpp"
#include <vector>
#include <string>
#include <variant>
#include <cstdint>

/*
 * Value — tagged union representing runtime values
 *
 * We use std::variant (C++17) for type safety.
 * Supported types:
 *   double  — all numbers (Lox-mini has no integer/float distinction)
 *   bool    — true/false
 *   std::monostate — nil (no value)
 *   std::string    — string values
 */
using Value = std::variant<std::monostate, double, bool, std::string>;

inline bool is_nil   (const Value &v) { return std::holds_alternative<std::monostate>(v); }
inline bool is_number(const Value &v) { return std::holds_alternative<double>(v); }
inline bool is_bool  (const Value &v) { return std::holds_alternative<bool>(v); }
inline bool is_string(const Value &v) { return std::holds_alternative<std::string>(v); }

inline double      as_number(const Value &v) { return std::get<double>(v); }
inline bool        as_bool  (const Value &v) { return std::get<bool>(v); }
inline std::string as_string(const Value &v) { return std::get<std::string>(v); }

/* Truthy: nil and false are falsy; everything else is truthy */
inline bool is_truthy(const Value &v) {
    if (is_nil(v))  return false;
    if (is_bool(v)) return as_bool(v);
    return true;
}

/* Value equality */
inline bool values_equal(const Value &a, const Value &b) {
    if (a.index() != b.index()) return false;
    if (is_nil(a)) return true;  // nil == nil
    if (is_number(a)) return as_number(a) == as_number(b);
    if (is_bool(a))   return as_bool(a)   == as_bool(b);
    if (is_string(a)) return as_string(a) == as_string(b);
    return false;
}

std::string value_to_string(const Value &v);

/*
 * Chunk — a compiled unit of bytecode
 *
 * Contains:
 *   - code: raw bytes (opcodes + operands)
 *   - constants: pool of constant Values (numbers, strings, identifiers)
 *   - lines: line number for each byte (for error messages)
 */
struct Chunk {
    std::vector<uint8_t>  code;       /* bytecode */
    std::vector<Value>    constants;  /* constant pool */
    std::vector<int>      lines;      /* line number per byte */

    /* Write one byte of bytecode */
    void write(uint8_t byte, int line) {
        code.push_back(byte);
        lines.push_back(line);
    }

    void write_opcode(OpCode op, int line) {
        write(static_cast<uint8_t>(op), line);
    }

    /* Add a constant to the pool and return its index */
    int add_constant(Value v) {
        constants.push_back(std::move(v));
        return (int)constants.size() - 1;
    }

    /* Disassemble for debugging */
    void disassemble(const std::string &name) const;
    int  disassemble_instruction(int offset) const;
};
