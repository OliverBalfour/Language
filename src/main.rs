
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
    tokens: Vec<Token<'a>>,

    _pos: usize,  // pos in source we're up to in tokenizer
    _token: usize,  // start pos of current token
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
        let mut stream = Self {
            source,
            tokens: Vec::with_capacity(source.len() / 10),
            _pos: 0,
            _token: 0,
        };
        stream.tokenize();
        stream
    }
    fn tokenize(&mut self) {
        loop {
            match self.next_token() {
                Some(token_type) => {
                    let eof = token_type == TokenType::EOF;
                    self.tokens.push(Token {
                        lexeme: self.lexeme(),
                        token: token_type,
                    });
                    if eof { break }
                },
                None => break
            }
        }
        if self._pos < self.source.len() {
            let snippet = &self.source[self._pos..std::cmp::min(self._pos+10, self.source.len())];
            BaseError::Syntax(format!("Unexpected characters: {}", snippet)).print()
        }
    }
    fn peek(&self) -> Option<char> {
        self.source[self._pos..].chars().nth(0)
    }
    fn test<F>(&mut self, pred: F) -> bool
        where F: FnOnce(char) -> bool
    {
        match self.peek() {
            Some(c) => pred(c),
            None => false,
        }
    }
    fn consume(&mut self, n: usize) {
        self._pos += n
    }
    fn skip(&mut self) {
        self._pos += 1;
        self._token = self._pos
    }
    fn lexeme(&self) -> &'a str {
        &self.source[self._token..self._pos]
    }
    fn rest(&self) -> &'a str {
        &self.source[self._token..]
    }
    fn next_token(&mut self) -> Option<TokenType> {
        // start scanning the new token from the end of the last token
        self._token = self._pos;

        // skip whitespace, handle EOFs
        while self.test(|c| c.is_whitespace()) { self.skip() }
        if let None = self.peek() { return Some(TokenType::EOF) }
        let c = self.peek().unwrap();

        // single character tokens
        self.consume(1);
        match c {
            '(' => return Some(TokenType::LeftParen),
            ')' => return Some(TokenType::RightParen),
            '+' => return Some(TokenType::Plus),
            '-' => return Some(TokenType::Minus),
            '*' => return Some(TokenType::Asterisk),
            '/' => return Some(TokenType::ForwardSlash),
            '=' => return Some(TokenType::Equals),
            ';' => return Some(TokenType::Semicolon),
            _ => self._pos -= 1  // un-consume
        }

        // natural numbers
        if c.is_numeric() {
            while self.test(|c| c.is_numeric()) { self.consume(1) }
            let n = self.lexeme().parse::<u64>().unwrap();
            return Some(TokenType::Natural(n));
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
            if self.rest().starts_with(lexeme) {
                self.consume(lexeme.len());
                return Some(token)
            }
        }

        // identifiers
        if c.is_ascii_alphabetic() || c == '_' {
            self.consume(1);
            while self.test(|c| c.is_ascii_alphanumeric() || c == '_') { self.consume(1) }
            return Some(TokenType::Identifier)
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
