#include "vm.hpp"
#include "compiler.hpp"
#include <iostream>
#include <sstream>
#include <cmath>

/* ============================================================
 * value_to_string — convert a Value to a printable string
 * ============================================================ */
std::string value_to_string(const Value &v) {
    if (is_nil(v))    return "nil";
    if (is_bool(v))   return as_bool(v) ? "true" : "false";
    if (is_string(v)) return as_string(v);
    if (is_number(v)) {
        double d = as_number(v);
        /* Print integers without decimal point */
        if (d == (long long)d && std::abs(d) < 1e15) {
            std::ostringstream oss;
            oss << (long long)d;
            return oss.str();
        }
        std::ostringstream oss;
        oss << d;
        return oss.str();
    }
    return "?";
}

/* ============================================================
 * Stack operations
 * ============================================================ */

void VM::push(Value v) {
    m_stack.push_back(std::move(v));
}

Value VM::pop() {
    if (m_stack.empty()) {
        runtime_error("Stack underflow");
        return std::monostate{};
    }
    Value v = std::move(m_stack.back());
    m_stack.pop_back();
    return v;
}

Value VM::peek(int distance) const {
    if ((int)m_stack.size() <= distance) return std::monostate{};
    return m_stack[m_stack.size() - 1 - distance];
}

/* ============================================================
 * Read helpers
 * ============================================================ */

uint8_t VM::read_byte() {
    return m_chunk.code[m_ip++];
}

uint16_t VM::read_short() {
    uint16_t hi = read_byte();
    uint16_t lo = read_byte();
    return (hi << 8) | lo;
}

Value VM::read_constant() {
    uint8_t idx = read_byte();
    return m_chunk.constants[idx];
}

/* ============================================================
 * Runtime error
 * ============================================================ */

void VM::runtime_error(const std::string &msg) {
    /* Find line number for current instruction */
    int line = (m_ip > 0 && m_ip - 1 < m_chunk.lines.size())
               ? m_chunk.lines[m_ip - 1] : -1;
    std::cerr << "[line " << line << "] Runtime error: " << msg << "\n";
    m_stack.clear();
}

/* ============================================================
 * interpret — entry point
 * ============================================================ */

InterpretResult VM::interpret(const std::string &source) {
    m_chunk   = Chunk{};
    m_ip      = 0;
    m_stack.clear();

    Compiler compiler;
    if (!compiler.compile(source, m_chunk)) {
        return InterpretResult::COMPILE_ERROR;
    }

    return run();
}

/* ============================================================
 * run — main interpreter loop (the "eval loop")
 * ============================================================ */

InterpretResult VM::run() {
#define BINARY_OP(opcode, op, result_type) \
    case OpCode::opcode: { \
        if (!is_number(peek(0)) || !is_number(peek(1))) { \
            runtime_error("Operands must be numbers"); \
            return InterpretResult::RUNTIME_ERROR; \
        } \
        double b = as_number(pop()); \
        double a = as_number(pop()); \
        push(result_type(a op b)); \
        break; \
    }

    while (true) {
        if (m_ip >= m_chunk.code.size()) break;

        OpCode instruction = (OpCode)read_byte();

        switch (instruction) {

        case OpCode::OP_CONST:
            push(read_constant());
            break;

        BINARY_OP(OP_ADD, +, double)
        BINARY_OP(OP_SUB, -, double)
        BINARY_OP(OP_MUL, *, double)

        case OpCode::OP_DIV: {
            if (!is_number(peek(0)) || !is_number(peek(1))) {
                runtime_error("Operands must be numbers");
                return InterpretResult::RUNTIME_ERROR;
            }
            double b = as_number(pop());
            double a = as_number(pop());
            if (b == 0.0) {
                runtime_error("Division by zero");
                return InterpretResult::RUNTIME_ERROR;
            }
            push(a / b);
            break;
        }

        case OpCode::OP_MOD: {
            if (!is_number(peek(0)) || !is_number(peek(1))) {
                runtime_error("Operands must be numbers");
                return InterpretResult::RUNTIME_ERROR;
            }
            double b = as_number(pop());
            double a = as_number(pop());
            if (b == 0.0) {
                runtime_error("Modulo by zero");
                return InterpretResult::RUNTIME_ERROR;
            }
            push(std::fmod(a, b));
            break;
        }

        case OpCode::OP_NEG: {
            if (!is_number(peek(0))) {
                runtime_error("Operand must be a number");
                return InterpretResult::RUNTIME_ERROR;
            }
            push(-as_number(pop()));
            break;
        }

        case OpCode::OP_NOT:
            push(!is_truthy(pop()));
            break;

        case OpCode::OP_EQ: {
            Value b = pop();
            Value a = pop();
            push(values_equal(a, b));
            break;
        }
        case OpCode::OP_NEQ: {
            Value b = pop();
            Value a = pop();
            push(!values_equal(a, b));
            break;
        }

        BINARY_OP(OP_LT, <,  bool)
        BINARY_OP(OP_GT, >,  bool)
        BINARY_OP(OP_LE, <=, bool)
        BINARY_OP(OP_GE, >=, bool)

        case OpCode::OP_PRINT: {
            Value v = pop();
            std::cout << value_to_string(v) << "\n";
            break;
        }

        case OpCode::OP_POP:
            pop();
            break;

        case OpCode::OP_DEFINE_GLOBAL: {
            std::string name = as_string(read_constant());
            m_globals[name]  = pop();
            break;
        }

        case OpCode::OP_GET_GLOBAL: {
            std::string name = as_string(read_constant());
            auto it = m_globals.find(name);
            if (it == m_globals.end()) {
                runtime_error("Undefined variable '" + name + "'");
                return InterpretResult::RUNTIME_ERROR;
            }
            push(it->second);
            break;
        }

        case OpCode::OP_SET_GLOBAL: {
            std::string name = as_string(read_constant());
            auto it = m_globals.find(name);
            if (it == m_globals.end()) {
                runtime_error("Undefined variable '" + name + "'");
                return InterpretResult::RUNTIME_ERROR;
            }
            it->second = peek(0);  /* peek, not pop — assignment is an expression */
            break;
        }

        case OpCode::OP_JUMP: {
            uint16_t offset = read_short();
            m_ip += offset;
            break;
        }

        case OpCode::OP_JUMP_IF_FALSE: {
            uint16_t offset = read_short();
            if (!is_truthy(peek(0))) m_ip += offset;
            break;
        }

        case OpCode::OP_LOOP: {
            uint16_t offset = read_short();
            m_ip -= offset;
            break;
        }

        case OpCode::OP_RETURN:
            return InterpretResult::OK;

        default:
            runtime_error("Unknown opcode");
            return InterpretResult::RUNTIME_ERROR;
        }
    }

#undef BINARY_OP
    return InterpretResult::OK;
}
