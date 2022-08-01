
use std::io::{self, BufRead};

#[derive(Debug, PartialEq)]
enum TokenType {
    LeftParen,
    RightParen,
    Plus,
    Minus,
    Asterisk,
    ForwardSlash,
    Equals,
    Semicolon,
    Natural(u64),
    If,
    Else,
    True,
    False,
    While,
    Return,
    Identifier,
    EOF,
}

#[derive(Debug)]
struct Token<'a> {
    lexeme: &'a str,  // slice into the source
    // line: u32,
    token: TokenType,
}

#[derive(Debug)]
struct TokenStream<'a> {
    source: &'a str,
    rest: &'a str, // non-empty implies syntax error
    tokens: Vec<Token<'a>>,
}

enum BaseError {
    Syntax(String),
}

impl BaseError {
    fn print(&self) {
        match self {
            BaseError::Syntax(x) => println!("SyntaxError: {}", x),
        }
    }
}

impl<'a> TokenStream<'a> {
    fn new(source: &'a str) -> Self {
        let mut tokens = Vec::with_capacity(source.len() / 10);
        // tokenize
        let mut rest = &source[..];
        loop {
            match Self::next_token(rest) {
                Some((token, skip)) => {
                    rest = &rest[skip + token.lexeme.len()..];
                    if token.token == TokenType::EOF { tokens.push(token); break }
                    tokens.push(token);
                },
                None => break
            }
        }
        if rest.len() > 0 {
            BaseError::Syntax(format!("Unexpected characters: {}", &rest[..std::cmp::min(rest.len(), 10)])).print()
        }
        Self {
            source,
            rest,
            tokens,
        }
    }
    fn next_token(_source: &'a str) -> Option<(Token<'a>, usize)> {
        // TODO: we could make this stateful and build peek, consume, consumeIf(pred), skipIf(pred) methods
        //   which would simplify most of these cases
        // skip over leading spaces
        let mut spaces: usize = 0;
        for c in _source.chars() {
            if c.is_whitespace() { spaces += 1 }
            else { break }
        }
        let source = &_source[spaces..];
        if source.len() == 0 {
            return Some((Token { lexeme: &source[0..0], token: TokenType::EOF }, spaces))
        }
        // single character tokens
        let c = source.chars().next()?;
        if let Some(token) = match c {
            '(' => Some(TokenType::LeftParen),
            ')' => Some(TokenType::RightParen),
            '+' => Some(TokenType::Plus),
            '-' => Some(TokenType::Minus),
            '*' => Some(TokenType::Asterisk),
            '/' => Some(TokenType::ForwardSlash),
            '=' => Some(TokenType::Equals),
            ';' => Some(TokenType::Semicolon),
            _ => None
        } { return Some((Token { lexeme: &source[0..1], token }, spaces)) }
        // natural numbers
        let mut lexeme = &source[0..0];
        for (i, c) in source.chars().enumerate() {
            if c.is_digit(10) { lexeme = &source[0..i+1] }
            else { break }
        }
        if lexeme.len() > 0 {
            return Some((Token { lexeme, token: TokenType::Natural(lexeme.parse::<u64>().unwrap()) }, spaces))
        }
        // reserved words
        const RESERVED: [(&str, TokenType); 6] = [
            ("if", TokenType::If),
            ("else", TokenType::Else),
            ("true", TokenType::True),
            ("false", TokenType::False),
            ("while", TokenType::While),
            ("return", TokenType::Return),
        ];
        for (lexeme, token) in RESERVED {
            if source.starts_with(lexeme) {
                return Some((Token { lexeme, token }, spaces))
            }
        }
        // identifiers
        let c = source.chars().next()?;
        if c.is_ascii_alphabetic() || c == '_' {
            let mut len = 1;
            for c in source[1..].chars() {
                if c.is_ascii_alphanumeric() || c == '-' { len += 1 }
                else { break }
            }
            return Some((Token { lexeme: &source[0..len], token: TokenType::Identifier }, spaces))
        }
        // failed
        None
    }
}

fn main() {
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        dbg!(TokenStream::new(&line.unwrap()).tokens);
    }
}
