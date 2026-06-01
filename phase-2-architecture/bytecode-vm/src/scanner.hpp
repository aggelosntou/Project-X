#pragma once
#include <string>

enum class TokenType {
    // Single character
    LPAREN, RPAREN, LBRACE, RBRACE,
    COMMA, DOT, MINUS, PLUS, SEMICOLON, SLASH, STAR, PERCENT,

    // One or two character
    BANG, BANG_EQ,
    EQ, EQ_EQ,
    GT, GT_EQ,
    LT, LT_EQ,

    // Literals
    IDENT, STRING, NUMBER,

    // Keywords
    AND, CLASS, ELSE, FALSE, FOR, FUN, IF, NIL,
    OR, PRINT, RETURN, SUPER, THIS, TRUE, VAR, WHILE,

    // Special
    ERROR, END_OF_FILE,
};

struct Token {
    TokenType   type;
    std::string lexeme;   /* source text of token */
    int         line;
};

class Scanner {
public:
    Scanner(const std::string &source);
    Token next_token();

private:
    std::string m_src;
    int         m_start;
    int         m_current;
    int         m_line;

    bool at_end() const;
    char advance();
    char peek() const;
    char peek_next() const;
    bool match(char expected);

    Token make_token(TokenType type) const;
    Token error_token(const std::string &msg) const;

    void skip_whitespace_and_comments();
    Token scan_string();
    Token scan_number();
    Token scan_identifier();

    TokenType identifier_type(const std::string &s) const;
};
