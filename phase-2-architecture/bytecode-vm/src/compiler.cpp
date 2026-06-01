#include "compiler.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

/* ============================================================
 * Parse rule table
 *
 * For each token type: {prefix_fn, infix_fn, precedence}
 * Indexed by (int)TokenType.
 *
 * We use a fixed-size array indexed by the enum's integer value.
 * The ORDER here must match the TokenType enum ORDER exactly.
 * ============================================================ */
const Compiler::ParseRule Compiler::rules[] = {
    /* LPAREN    */ {&Compiler::grouping, nullptr,          PREC_NONE},
    /* RPAREN    */ {nullptr,             nullptr,          PREC_NONE},
    /* LBRACE    */ {nullptr,             nullptr,          PREC_NONE},
    /* RBRACE    */ {nullptr,             nullptr,          PREC_NONE},
    /* COMMA     */ {nullptr,             nullptr,          PREC_NONE},
    /* DOT       */ {nullptr,             nullptr,          PREC_NONE},
    /* MINUS     */ {&Compiler::unary,    &Compiler::binary, PREC_TERM},
    /* PLUS      */ {nullptr,             &Compiler::binary, PREC_TERM},
    /* SEMICOLON */ {nullptr,             nullptr,          PREC_NONE},
    /* SLASH     */ {nullptr,             &Compiler::binary, PREC_FACTOR},
    /* STAR      */ {nullptr,             &Compiler::binary, PREC_FACTOR},
    /* PERCENT   */ {nullptr,             &Compiler::binary, PREC_FACTOR},
    /* BANG      */ {&Compiler::unary,    nullptr,          PREC_NONE},
    /* BANG_EQ   */ {nullptr,             &Compiler::binary, PREC_EQUALITY},
    /* EQ        */ {nullptr,             nullptr,          PREC_NONE},
    /* EQ_EQ     */ {nullptr,             &Compiler::binary, PREC_EQUALITY},
    /* GT        */ {nullptr,             &Compiler::binary, PREC_COMPARISON},
    /* GT_EQ     */ {nullptr,             &Compiler::binary, PREC_COMPARISON},
    /* LT        */ {nullptr,             &Compiler::binary, PREC_COMPARISON},
    /* LT_EQ     */ {nullptr,             &Compiler::binary, PREC_COMPARISON},
    /* IDENT     */ {&Compiler::variable, nullptr,          PREC_NONE},
    /* STRING    */ {&Compiler::string_lit, nullptr,        PREC_NONE},
    /* NUMBER    */ {&Compiler::number,   nullptr,          PREC_NONE},
    /* AND       */ {nullptr,             &Compiler::and_op, PREC_AND},
    /* CLASS     */ {nullptr,             nullptr,          PREC_NONE},
    /* ELSE      */ {nullptr,             nullptr,          PREC_NONE},
    /* FALSE     */ {&Compiler::literal,  nullptr,          PREC_NONE},
    /* FOR       */ {nullptr,             nullptr,          PREC_NONE},
    /* FUN       */ {nullptr,             nullptr,          PREC_NONE},
    /* IF        */ {nullptr,             nullptr,          PREC_NONE},
    /* NIL       */ {&Compiler::literal,  nullptr,          PREC_NONE},
    /* OR        */ {nullptr,             &Compiler::or_op,  PREC_OR},
    /* PRINT     */ {nullptr,             nullptr,          PREC_NONE},
    /* RETURN    */ {nullptr,             nullptr,          PREC_NONE},
    /* SUPER     */ {nullptr,             nullptr,          PREC_NONE},
    /* THIS      */ {nullptr,             nullptr,          PREC_NONE},
    /* TRUE      */ {&Compiler::literal,  nullptr,          PREC_NONE},
    /* VAR       */ {nullptr,             nullptr,          PREC_NONE},
    /* WHILE     */ {nullptr,             nullptr,          PREC_NONE},
    /* ERROR     */ {nullptr,             nullptr,          PREC_NONE},
    /* END_OF_FILE */ {nullptr,           nullptr,          PREC_NONE},
};

const Compiler::ParseRule &Compiler::get_rule(TokenType type) const {
    return rules[static_cast<int>(type)];
}

/* ============================================================
 * Parsing primitives
 * ============================================================ */

void Compiler::advance() {
    m_previous = m_current;
    while (true) {
        m_current = m_scanner->next_token();
        if (m_current.type != TokenType::ERROR) break;
        error_at_current(m_current.lexeme);
    }
}

void Compiler::consume(TokenType type, const std::string &msg) {
    if (m_current.type == type) { advance(); return; }
    error_at_current(msg);
}

bool Compiler::check(TokenType type) const {
    return m_current.type == type;
}

bool Compiler::match_tok(TokenType type) {
    if (!check(type)) return false;
    advance();
    return true;
}

void Compiler::error_at(const Token &tok, const std::string &msg) {
    if (m_panic_mode) return;
    m_panic_mode = true;
    m_had_error  = true;
    std::cerr << "[line " << tok.line << "] Error";
    if (tok.type == TokenType::END_OF_FILE) std::cerr << " at end";
    else std::cerr << " at '" << tok.lexeme << "'";
    std::cerr << ": " << msg << "\n";
}

void Compiler::error_at_current(const std::string &msg) { error_at(m_current, msg); }
void Compiler::error_at_previous(const std::string &msg) { error_at(m_previous, msg); }

void Compiler::synchronize() {
    m_panic_mode = false;
    while (m_current.type != TokenType::END_OF_FILE) {
        if (m_previous.type == TokenType::SEMICOLON) return;
        switch (m_current.type) {
        case TokenType::CLASS: case TokenType::FUN:
        case TokenType::VAR:   case TokenType::FOR:
        case TokenType::IF:    case TokenType::WHILE:
        case TokenType::PRINT: case TokenType::RETURN:
            return;
        default: break;
        }
        advance();
    }
}

/* ============================================================
 * Code emission
 * ============================================================ */

void Compiler::emit_byte(uint8_t byte) {
    m_chunk->write(byte, m_previous.line);
}

void Compiler::emit_opcode(OpCode op) {
    emit_byte(static_cast<uint8_t>(op));
}

void Compiler::emit_two(uint8_t a, uint8_t b) {
    emit_byte(a); emit_byte(b);
}

void Compiler::emit_return() {
    emit_opcode(OpCode::OP_RETURN);
}

int Compiler::emit_jump(OpCode op) {
    emit_opcode(op);
    emit_byte(0xFF);  /* placeholder — patched later */
    emit_byte(0xFF);
    return (int)m_chunk->code.size() - 2;  /* offset of the placeholder */
}

void Compiler::patch_jump(int offset) {
    /* Jump size = current position - (offset + 2) */
    int jump = (int)m_chunk->code.size() - offset - 2;
    if (jump > 65535) {
        error_at_previous("Jump too large");
        return;
    }
    m_chunk->code[offset]     = (uint8_t)((jump >> 8) & 0xFF);
    m_chunk->code[offset + 1] = (uint8_t)(jump & 0xFF);
}

void Compiler::emit_loop(int loop_start) {
    emit_opcode(OpCode::OP_LOOP);
    int offset = (int)m_chunk->code.size() - loop_start + 2;
    if (offset > 65535) error_at_previous("Loop too large");
    emit_byte((uint8_t)((offset >> 8) & 0xFF));
    emit_byte((uint8_t)(offset & 0xFF));
}

int Compiler::make_constant(Value v) {
    int idx = m_chunk->add_constant(std::move(v));
    if (idx > 255) { error_at_previous("Too many constants in one chunk"); return 0; }
    return idx;
}

void Compiler::emit_constant(Value v) {
    emit_two(static_cast<uint8_t>(OpCode::OP_CONST), (uint8_t)make_constant(std::move(v)));
}

/* ============================================================
 * Statements
 * ============================================================ */

bool Compiler::compile(const std::string &source, Chunk &chunk) {
    Scanner scanner(source);
    m_scanner    = &scanner;
    m_chunk      = &chunk;
    m_had_error  = false;
    m_panic_mode = false;

    advance();  /* prime the parser */

    while (!check(TokenType::END_OF_FILE)) {
        declaration();
    }

    emit_return();
    return !m_had_error;
}

void Compiler::declaration() {
    if (match_tok(TokenType::VAR)) {
        var_declaration();
    } else {
        statement();
    }
    if (m_panic_mode) synchronize();
}

void Compiler::var_declaration() {
    consume(TokenType::IDENT, "Expected variable name");
    std::string name = m_previous.lexeme;
    int name_const   = identifier_constant(name);

    if (match_tok(TokenType::EQ)) {
        expression();
    } else {
        emit_opcode(OpCode::OP_CONST);
        emit_byte((uint8_t)make_constant(std::monostate{}));  /* nil default */
    }

    consume(TokenType::SEMICOLON, "Expected ';' after variable declaration");
    emit_two(static_cast<uint8_t>(OpCode::OP_DEFINE_GLOBAL), (uint8_t)name_const);
}

void Compiler::statement() {
    if (match_tok(TokenType::PRINT)) {
        print_statement();
    } else if (match_tok(TokenType::IF)) {
        if_statement();
    } else if (match_tok(TokenType::WHILE)) {
        while_statement();
    } else if (match_tok(TokenType::LBRACE)) {
        block();
    } else {
        expression_statement();
    }
}

void Compiler::print_statement() {
    expression();
    consume(TokenType::SEMICOLON, "Expected ';' after value");
    emit_opcode(OpCode::OP_PRINT);
}

void Compiler::expression_statement() {
    expression();
    consume(TokenType::SEMICOLON, "Expected ';' after expression");
    emit_opcode(OpCode::OP_POP);
}

void Compiler::if_statement() {
    consume(TokenType::LPAREN, "Expected '(' after 'if'");
    expression();
    consume(TokenType::RPAREN, "Expected ')' after condition");

    int then_jump = emit_jump(OpCode::OP_JUMP_IF_FALSE);
    emit_opcode(OpCode::OP_POP);  /* pop condition */

    statement();

    int else_jump = emit_jump(OpCode::OP_JUMP);
    patch_jump(then_jump);
    emit_opcode(OpCode::OP_POP);  /* pop condition (else branch) */

    if (match_tok(TokenType::ELSE)) {
        statement();
    }
    patch_jump(else_jump);
}

void Compiler::while_statement() {
    int loop_start = (int)m_chunk->code.size();

    consume(TokenType::LPAREN, "Expected '(' after 'while'");
    expression();
    consume(TokenType::RPAREN, "Expected ')' after condition");

    int exit_jump = emit_jump(OpCode::OP_JUMP_IF_FALSE);
    emit_opcode(OpCode::OP_POP);  /* pop condition */

    statement();

    emit_loop(loop_start);

    patch_jump(exit_jump);
    emit_opcode(OpCode::OP_POP);  /* pop condition */
}

void Compiler::block() {
    while (!check(TokenType::RBRACE) && !check(TokenType::END_OF_FILE)) {
        declaration();
    }
    consume(TokenType::RBRACE, "Expected '}' after block");
}

/* ============================================================
 * Expressions — Pratt parser
 * ============================================================ */

void Compiler::expression() {
    parse_precedence(PREC_ASSIGNMENT);
}

void Compiler::parse_precedence(Precedence prec) {
    advance();
    ParseFn prefix_rule = get_rule(m_previous.type).prefix;
    if (!prefix_rule) {
        error_at_previous("Expected expression");
        return;
    }

    bool can_assign = (prec <= PREC_ASSIGNMENT);
    (this->*prefix_rule)(can_assign);

    while (prec <= get_rule(m_current.type).precedence) {
        advance();
        ParseFn infix_rule = get_rule(m_previous.type).infix;
        (this->*infix_rule)(can_assign);
    }

    if (can_assign && match_tok(TokenType::EQ)) {
        error_at_previous("Invalid assignment target");
    }
}

/* ---- Prefix parse functions ---- */

void Compiler::number(bool /*can_assign*/) {
    double val = std::stod(m_previous.lexeme);
    emit_constant(val);
}

void Compiler::string_lit(bool /*can_assign*/) {
    emit_constant(m_previous.lexeme);
}

void Compiler::literal(bool /*can_assign*/) {
    switch (m_previous.type) {
    case TokenType::TRUE:  emit_constant(true);              break;
    case TokenType::FALSE: emit_constant(false);             break;
    case TokenType::NIL:   emit_constant(std::monostate{}); break;
    default: break;
    }
}

void Compiler::grouping(bool /*can_assign*/) {
    expression();
    consume(TokenType::RPAREN, "Expected ')' after expression");
}

void Compiler::unary(bool /*can_assign*/) {
    TokenType op = m_previous.type;
    parse_precedence(PREC_UNARY);
    if (op == TokenType::MINUS) emit_opcode(OpCode::OP_NEG);
    else if (op == TokenType::BANG) emit_opcode(OpCode::OP_NOT);
}

void Compiler::variable(bool can_assign) {
    named_variable(m_previous.lexeme, can_assign);
}

/* ---- Infix parse functions ---- */

void Compiler::binary(bool /*can_assign*/) {
    TokenType op = m_previous.type;
    const ParseRule &rule = get_rule(op);
    /* Parse right operand at one higher precedence (left-associative) */
    parse_precedence((Precedence)((int)rule.precedence + 1));

    switch (op) {
    case TokenType::PLUS:    emit_opcode(OpCode::OP_ADD); break;
    case TokenType::MINUS:   emit_opcode(OpCode::OP_SUB); break;
    case TokenType::STAR:    emit_opcode(OpCode::OP_MUL); break;
    case TokenType::SLASH:   emit_opcode(OpCode::OP_DIV); break;
    case TokenType::PERCENT: emit_opcode(OpCode::OP_MOD); break;
    case TokenType::EQ_EQ:   emit_opcode(OpCode::OP_EQ);  break;
    case TokenType::BANG_EQ: emit_opcode(OpCode::OP_NEQ); break;
    case TokenType::LT:      emit_opcode(OpCode::OP_LT);  break;
    case TokenType::GT:      emit_opcode(OpCode::OP_GT);  break;
    case TokenType::LT_EQ:   emit_opcode(OpCode::OP_LE);  break;
    case TokenType::GT_EQ:   emit_opcode(OpCode::OP_GE);  break;
    default: break;
    }
}

void Compiler::and_op(bool /*can_assign*/) {
    /* Short-circuit: if left is false, skip right */
    int end_jump = emit_jump(OpCode::OP_JUMP_IF_FALSE);
    emit_opcode(OpCode::OP_POP);
    parse_precedence(PREC_AND);
    patch_jump(end_jump);
}

void Compiler::or_op(bool /*can_assign*/) {
    /* Short-circuit: if left is true, skip right */
    int else_jump = emit_jump(OpCode::OP_JUMP_IF_FALSE);
    int end_jump  = emit_jump(OpCode::OP_JUMP);
    patch_jump(else_jump);
    emit_opcode(OpCode::OP_POP);
    parse_precedence(PREC_OR);
    patch_jump(end_jump);
}

/* ---- Variable helpers ---- */

int Compiler::identifier_constant(const std::string &name) {
    return make_constant(name);
}

void Compiler::named_variable(const std::string &name, bool can_assign) {
    int arg = identifier_constant(name);
    if (can_assign && match_tok(TokenType::EQ)) {
        expression();
        emit_two(static_cast<uint8_t>(OpCode::OP_SET_GLOBAL), (uint8_t)arg);
    } else {
        emit_two(static_cast<uint8_t>(OpCode::OP_GET_GLOBAL), (uint8_t)arg);
    }
}
