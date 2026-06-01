#pragma once
#include "chunk.hpp"
#include "scanner.hpp"
#include <string>
#include <memory>

/*
 * Compiler — single-pass Pratt parser that emits bytecode directly
 *
 * A Pratt parser associates a "parse function" and a "precedence level"
 * with each token type. Expression parsing is driven by precedence:
 *   parse_precedence(PREC_ASSIGNMENT) handles the full expression.
 *
 * For each token, we look up:
 *   - prefix_fn:  how to parse it as the START of an expression (e.g., number, unary -)
 *   - infix_fn:   how to parse it in the MIDDLE of an expression (e.g., +, *, ==)
 *   - precedence: how tightly it binds (determines when to stop recursing)
 */

class Compiler {
public:
    /*
     * compile — compile source text into a Chunk
     * Returns false if a compile error occurred.
     */
    bool compile(const std::string &source, Chunk &chunk);

private:
    /* Parser state */
    Scanner *m_scanner  = nullptr;
    Chunk   *m_chunk    = nullptr;
    Token    m_current;
    Token    m_previous;
    bool     m_had_error    = false;
    bool     m_panic_mode   = false;

    /* ---- Parsing primitives ---- */
    void advance();
    void consume(TokenType type, const std::string &msg);
    bool check(TokenType type) const;
    bool match_tok(TokenType type);
    void error_at_current(const std::string &msg);
    void error_at_previous(const std::string &msg);
    void error_at(const Token &tok, const std::string &msg);
    void synchronize();

    /* ---- Code emission ---- */
    void emit_byte(uint8_t byte);
    void emit_opcode(OpCode op);
    void emit_two(uint8_t a, uint8_t b);
    void emit_return();
    int  emit_jump(OpCode op);
    void patch_jump(int offset);
    void emit_loop(int loop_start);
    int  make_constant(Value v);
    void emit_constant(Value v);

    /* ---- Statements ---- */
    void declaration();
    void var_declaration();
    void statement();
    void print_statement();
    void expression_statement();
    void if_statement();
    void while_statement();
    void block();

    /* ---- Expressions (Pratt parser) ---- */
    enum Precedence {
        PREC_NONE,
        PREC_ASSIGNMENT,  /* = */
        PREC_OR,          /* or */
        PREC_AND,         /* and */
        PREC_EQUALITY,    /* == != */
        PREC_COMPARISON,  /* < > <= >= */
        PREC_TERM,        /* + - */
        PREC_FACTOR,      /* * / % */
        PREC_UNARY,       /* ! - */
        PREC_CALL,        /* . () */
        PREC_PRIMARY,
    };

    using ParseFn = void (Compiler::*)(bool can_assign);

    struct ParseRule {
        ParseFn    prefix;
        ParseFn    infix;
        Precedence precedence;
    };

    static const ParseRule rules[];
    const ParseRule &get_rule(TokenType type) const;

    void parse_precedence(Precedence prec);
    void expression();

    /* Parse functions (prefix) */
    void number(bool can_assign);
    void string_lit(bool can_assign);
    void literal(bool can_assign);
    void grouping(bool can_assign);
    void unary(bool can_assign);
    void variable(bool can_assign);

    /* Parse functions (infix) */
    void binary(bool can_assign);
    void and_op(bool can_assign);
    void or_op(bool can_assign);

    /* Variable helpers */
    int  identifier_constant(const std::string &name);
    void named_variable(const std::string &name, bool can_assign);
};
