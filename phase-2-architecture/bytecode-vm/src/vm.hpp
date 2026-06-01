#pragma once
#include "chunk.hpp"
#include <unordered_map>
#include <string>
#include <vector>

enum class InterpretResult {
    OK,
    COMPILE_ERROR,
    RUNTIME_ERROR,
};

class VM {
public:
    VM() = default;

    /* Compile and run source text */
    InterpretResult interpret(const std::string &source);

private:
    Chunk  m_chunk;
    size_t m_ip = 0;  /* instruction pointer (index into m_chunk.code) */

    /* Value stack (max 256 deep) */
    std::vector<Value> m_stack;

    /* Global variable table: name → value */
    std::unordered_map<std::string, Value> m_globals;

    /* Stack operations */
    void  push(Value v);
    Value pop();
    Value peek(int distance = 0) const;

    /* Run the bytecode */
    InterpretResult run();

    /* Runtime error reporting */
    void runtime_error(const std::string &msg);

    /* Read helpers */
    uint8_t  read_byte();
    uint16_t read_short();
    Value    read_constant();
};
