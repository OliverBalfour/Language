
#![allow(dead_code)]

use std::io::{self, BufRead};

#[derive(Debug, PartialEq, Clone, Copy)]
enum TokenType {
    LeftParen,
    RightParen,
    Plus,
    Minus,
    Asterisk,
    ForwardSlash,
    Semicolon,

    Equal,
    EqualEqual,
    Not,
    NotEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,

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

impl ToString for TokenType {
    fn to_string(&self) -> String {
        match self {
            Self::LeftParen => String::from("("),
            Self::RightParen => String::from(")"),
            Self::Plus => String::from("+"),
            Self::Minus => String::from("-"),
            Self::Asterisk => String::from("*"),
            Self::ForwardSlash => String::from("/"),
            Self::Semicolon => String::from(";"),
            Self::Equal => String::from("="),
            Self::EqualEqual => String::from("=="),
            Self::Not => String::from("!"),
            Self::NotEqual => String::from("!="),
            Self::Greater => String::from(">"),
            Self::GreaterEqual => String::from(">="),
            Self::Less => String::from("<"),
            Self::LessEqual => String::from("<="),
            _ => String::from(""),
        }
    }
}

// Token with associated line/source info
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

        // two character operators
        if self.rest().len() >= 2 {
            let op = &self.rest()[..2];
            self.consume(2);
            match op {
                "==" => return Some(TokenType::EqualEqual),
                "!=" => return Some(TokenType::NotEqual),
                ">=" => return Some(TokenType::GreaterEqual),
                "<=" => return Some(TokenType::LessEqual),
                _ => self._pos -= 2 // un-consume
            }
        }

        // single character tokens
        self.consume(1);
        match c {
            '(' => return Some(TokenType::LeftParen),
            ')' => return Some(TokenType::RightParen),
            '+' => return Some(TokenType::Plus),
            '-' => return Some(TokenType::Minus),
            '*' => return Some(TokenType::Asterisk),
            '/' => return Some(TokenType::ForwardSlash),
            ';' => return Some(TokenType::Semicolon),
            '=' => return Some(TokenType::Equal),
            '!' => return Some(TokenType::Not),
            '>' => return Some(TokenType::Greater),
            '<' => return Some(TokenType::Less),
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

#[derive(Debug)]
enum Expr {
    PrefixUnary { op: TokenType, expr: Box<Expr> },
    InfixBinary { left: Box<Expr>, op: TokenType, right: Box<Expr> },
    Natural(u64),
    Bool(bool),
}

impl ToString for Expr {
    fn to_string(&self) -> String {
        match self {
            Expr::PrefixUnary { op, expr } => format!("{}{}", op.to_string(), expr.to_string()),
            Expr::InfixBinary { left, op, right } => format!("({} {} {})", left.to_string(), op.to_string(), right.to_string()),
            Expr::Natural(n) => n.to_string(),
            Expr::Bool(b) => b.to_string(),
        }
    }
}

struct Parser<'a> {
    stream: TokenStream<'a>,
    root: Option<Expr>,
    _pos: usize,
}

impl<'a> Parser<'a> {
    fn new(stream: TokenStream<'a>) -> Self {
        let mut p = Self {
            stream,
            root: None,
            _pos: 0,
        };
        // parse root as expression
        p.root = p.expr();
        p
    }
    fn peek(&self) -> Option<TokenType> {
        Some(self.stream.tokens[self._pos..].iter().nth(0)?.token)
    }
    fn consume(&mut self) -> Option<TokenType> {
        let t = self.peek();
        self._pos += 1;
        t
    }
    // expr ::= equality
    fn expr(&mut self) -> Option<Expr> {
        self.equality()
    }
    // equality ::= comparison ( (!= | ==) comparison )*
    fn equality(&mut self) -> Option<Expr> {
        let mut expr = self.comparison()?;
        while self.peek() == Some(TokenType::NotEqual) || self.peek() == Some(TokenType::EqualEqual) {
            let op = self.consume().unwrap();
            let right = Box::new(self.comparison()?);
            expr = Expr::InfixBinary {
                left: Box::new(expr),
                op, right,
            }
        };
        Some(expr)
    }
    // comparison ::= term ( (> | >= | < | <=) term )*
    fn comparison(&mut self) -> Option<Expr> {
        let mut expr = self.term()?;
        while self.peek() == Some(TokenType::Greater) || self.peek() == Some(TokenType::GreaterEqual)
           || self.peek() == Some(TokenType::Less) || self.peek() == Some(TokenType::LessEqual) {
            let op = self.consume().unwrap();
            let right = Box::new(self.term()?);
            expr = Expr::InfixBinary {
                left: Box::new(expr),
                op, right,
            }
        };
        Some(expr)
    }
    // term ::= factor ( (- | +) factor )*
    fn term(&mut self) -> Option<Expr> {
        let mut expr = self.factor()?;
        while self.peek() == Some(TokenType::Minus) || self.peek() == Some(TokenType::Plus) {
            let op = self.consume().unwrap();
            let right = Box::new(self.factor()?);
            expr = Expr::InfixBinary {
                left: Box::new(expr),
                op, right,
            }
        };
        Some(expr)
    }
    // factor ::= unary ( ( / | * ) unary )*
    fn factor(&mut self) -> Option<Expr> {
        let mut expr = self.unary()?;
        while self.peek() == Some(TokenType::ForwardSlash) || self.peek() == Some(TokenType::Asterisk) {
            let op = self.consume().unwrap();
            let right = Box::new(self.unary()?);
            expr = Expr::InfixBinary {
                left: Box::new(expr),
                op, right,
            }
        };
        Some(expr)
    }
    // unary ::= ( "!" | "-" ) unary | primary
    fn unary(&mut self) -> Option<Expr> {
        if self.peek() == Some(TokenType::Not) || self.peek() == Some(TokenType::Minus) {
            // recursive case
            let op = self.consume().unwrap();
            let expr = Box::new(self.unary()?);
            Some(Expr::PrefixUnary { op, expr })
        } else {
            // base case
            self.primary()
        }
    }
    // primary ::= NUMBER | STRING | true | false | "(" expr ")"
    fn primary(&mut self) -> Option<Expr> {
        match self.consume()? {
            TokenType::Natural(n) => Some(Expr::Natural(n)),
            // TokenType::String(s) => Some(Expr::String(s)),
            TokenType::True => Some(Expr::Bool(true)),
            TokenType::False => Some(Expr::Bool(false)),
            TokenType::Identifier => panic!("help!"),
            TokenType::LeftParen => {
                let expr = self.expr()?;
                if self.consume() == Some(TokenType::RightParen) {
                    Some(expr)
                } else { None }
            },
            _ => None,
        }
    }
}

fn main() {
    let stdin = io::stdin();
    for _line in stdin.lock().lines() {
        let line = _line.unwrap();
        let stream = TokenStream::new(&line);
        let parser = Parser::new(stream);
        // dbg!(parser.stream.tokens);
        // dbg!(parser.root);
        println!("{}", parser.root.unwrap().to_string());
    }
}
