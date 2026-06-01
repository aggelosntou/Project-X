#include "scanner.hpp"
#include <cctype>
#include <unordered_map>

Scanner::Scanner(const std::string &source)
    : m_src(source), m_start(0), m_current(0), m_line(1) {}

bool Scanner::at_end() const {
    return m_current >= (int)m_src.size();
}

char Scanner::advance() {
    return m_src[m_current++];
}

char Scanner::peek() const {
    if (at_end()) return '\0';
    return m_src[m_current];
}

char Scanner::peek_next() const {
    if (m_current + 1 >= (int)m_src.size()) return '\0';
    return m_src[m_current + 1];
}

bool Scanner::match(char expected) {
    if (at_end()) return false;
    if (m_src[m_current] != expected) return false;
    m_current++;
    return true;
}

Token Scanner::make_token(TokenType type) const {
    return Token{ type, m_src.substr(m_start, m_current - m_start), m_line };
}

Token Scanner::error_token(const std::string &msg) const {
    return Token{ TokenType::ERROR, msg, m_line };
}

void Scanner::skip_whitespace_and_comments() {
    while (true) {
        char c = peek();
        switch (c) {
        case ' ':
        case '\r':
        case '\t':
            advance();
            break;
        case '\n':
            m_line++;
            advance();
            break;
        case '/':
            if (peek_next() == '/') {
                /* Line comment: skip to end of line */
                while (peek() != '\n' && !at_end()) advance();
            } else {
                return;
            }
            break;
        default:
            return;
        }
    }
}

Token Scanner::scan_string() {
    /* Opening " already consumed */
    while (peek() != '"' && !at_end()) {
        if (peek() == '\n') m_line++;
        advance();
    }
    if (at_end()) return error_token("Unterminated string");
    advance();  /* closing " */

    /* Strip surrounding quotes from lexeme */
    std::string content = m_src.substr(m_start + 1, m_current - m_start - 2);
    return Token{ TokenType::STRING, content, m_line };
}

Token Scanner::scan_number() {
    while (std::isdigit((unsigned char)peek())) advance();
    if (peek() == '.' && std::isdigit((unsigned char)peek_next())) {
        advance();  /* consume '.' */
        while (std::isdigit((unsigned char)peek())) advance();
    }
    return make_token(TokenType::NUMBER);
}

TokenType Scanner::identifier_type(const std::string &s) const {
    static const std::unordered_map<std::string, TokenType> keywords = {
        {"and",    TokenType::AND},
        {"class",  TokenType::CLASS},
        {"else",   TokenType::ELSE},
        {"false",  TokenType::FALSE},
        {"for",    TokenType::FOR},
        {"fun",    TokenType::FUN},
        {"if",     TokenType::IF},
        {"nil",    TokenType::NIL},
        {"or",     TokenType::OR},
        {"print",  TokenType::PRINT},
        {"return", TokenType::RETURN},
        {"super",  TokenType::SUPER},
        {"this",   TokenType::THIS},
        {"true",   TokenType::TRUE},
        {"var",    TokenType::VAR},
        {"while",  TokenType::WHILE},
    };
    auto it = keywords.find(s);
    return it != keywords.end() ? it->second : TokenType::IDENT;
}

Token Scanner::scan_identifier() {
    while (std::isalnum((unsigned char)peek()) || peek() == '_') advance();
    std::string lexeme = m_src.substr(m_start, m_current - m_start);
    return Token{ identifier_type(lexeme), lexeme, m_line };
}

Token Scanner::next_token() {
    skip_whitespace_and_comments();
    m_start = m_current;

    if (at_end()) return make_token(TokenType::END_OF_FILE);

    char c = advance();

    if (std::isalpha((unsigned char)c) || c == '_') return scan_identifier();
    if (std::isdigit((unsigned char)c))              return scan_number();

    switch (c) {
    case '(': return make_token(TokenType::LPAREN);
    case ')': return make_token(TokenType::RPAREN);
    case '{': return make_token(TokenType::LBRACE);
    case '}': return make_token(TokenType::RBRACE);
    case ',': return make_token(TokenType::COMMA);
    case '.': return make_token(TokenType::DOT);
    case '-': return make_token(TokenType::MINUS);
    case '+': return make_token(TokenType::PLUS);
    case ';': return make_token(TokenType::SEMICOLON);
    case '/': return make_token(TokenType::SLASH);
    case '*': return make_token(TokenType::STAR);
    case '%': return make_token(TokenType::PERCENT);
    case '!': return make_token(match('=') ? TokenType::BANG_EQ : TokenType::BANG);
    case '=': return make_token(match('=') ? TokenType::EQ_EQ   : TokenType::EQ);
    case '<': return make_token(match('=') ? TokenType::LT_EQ   : TokenType::LT);
    case '>': return make_token(match('=') ? TokenType::GT_EQ   : TokenType::GT);
    case '"': return scan_string();
    default:
        return error_token(std::string("Unexpected character '") + c + "'");
    }
}
