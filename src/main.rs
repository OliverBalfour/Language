
#![allow(dead_code)]

use std::io::{self, BufRead};
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Debug, PartialEq, Clone)]
enum TokenType {
    LeftParen,
    RightParen,
    LeftCurly,
    RightCurly,
    Plus,
    Minus,
    Asterisk,
    ForwardSlash,
    Semicolon,
    Comma,

    Equal,
    EqualEqual,
    Not,
    NotEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    And,
    Or,

    Natural(u64),
    String(Rc<String>),

    Fn,
    If,
    Else,
    While,
    True,
    False,
    Return,
    Print,
    Var,

    Identifier(String),

    Comment,
    EOF,
}

impl TokenType {
    fn fixity(&self) -> u8 {
        match self {
            TokenType::Asterisk => 70,
            TokenType::ForwardSlash => 70,
            TokenType::Plus => 60,
            TokenType::Minus => 60,
            TokenType::Greater => 50,
            TokenType::GreaterEqual => 50,
            TokenType::Less => 50,
            TokenType::LessEqual => 50,
            TokenType::EqualEqual => 40,
            TokenType::NotEqual => 40,
            TokenType::And => 30,
            TokenType::Or => 20,
            _ => 0,
        }
    }
}

impl ToString for TokenType {
    fn to_string(&self) -> String {
        match self {
            Self::LeftParen => String::from("("),
            Self::RightParen => String::from(")"),
            Self::LeftCurly => String::from("{"),
            Self::RightCurly => String::from("}"),
            Self::Plus => String::from("+"),
            Self::Minus => String::from("-"),
            Self::Asterisk => String::from("*"),
            Self::ForwardSlash => String::from("/"),
            Self::Semicolon => String::from(";"),
            Self::Comma => String::from(","),
            Self::Equal => String::from("="),
            Self::EqualEqual => String::from("=="),
            Self::Not => String::from("!"),
            Self::NotEqual => String::from("!="),
            Self::Greater => String::from(">"),
            Self::GreaterEqual => String::from(">="),
            Self::Less => String::from("<"),
            Self::LessEqual => String::from("<="),
            Self::And => String::from("&&"),
            Self::Or => String::from("||"),
            Self::Natural(n) => format!("{}", n),
            Self::String(s) => format!("\"{}\"", s.clone()),
            Self::Fn => String::from("fn"),
            Self::If => String::from("if"),
            Self::Else => String::from("else"),
            Self::While => String::from("while"),
            Self::True => String::from("true"),
            Self::False => String::from("false"),
            Self::Return => String::from("return "),
            Self::Print => String::from("print "),
            Self::Var => String::from("var "),
            Self::Identifier(s) => (s as &String).clone(),
            Self::EOF => String::from(" EOF"),
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
    err: Option<BaseError>,

    _pos: usize,  // pos in source we're up to in tokenizer
    _token: usize,  // start pos of current token
}

#[derive(Debug, Clone)]
enum BaseError {
    Unknown,
    NotImplemented,
    Syntax(String),
    Type(String),
    Name(String),
}

impl ToString for BaseError {
    fn to_string(&self) -> String {
        match self {
            BaseError::Unknown => String::from("UnknownError"),
            BaseError::NotImplemented => String::from("NotImplementedError"),
            BaseError::Syntax(x) => String::from(format!("SyntaxError: {}", x)),
            BaseError::Type(x) => String::from(format!("TypeError: {}", x)),
            BaseError::Name(x) => String::from(format!("NameError: {}", x)),
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
            err: None,
        };
        stream.tokenize();
        stream
    }
    fn tokenize(&mut self) {
        loop {
            match self.next_token() {
                Some(token_type) => {
                    let eof = token_type == TokenType::EOF;
                    let skip = token_type == TokenType::Comment;
                    if !skip {
                        self.tokens.push(Token {
                            lexeme: self.lexeme(),
                            token: token_type,
                        })
                    }
                    if eof { break }
                },
                None => break
            }
        }
        if self._pos < self.source.len() {
            let snippet = &self.source[self._pos..std::cmp::min(self._pos+10, self.source.len())];
            self.err = Some(BaseError::Syntax(format!("Unexpected characters: {}", snippet)))
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
                "&&" => return Some(TokenType::And),
                "||" => return Some(TokenType::Or),
                "//" => {
                    while self.test(|c| c != '\n') { self.consume(1) }
                    if self.peek() == Some('\n') { self.consume(1) }
                    return Some(TokenType::Comment)
                },
                _ => self._pos -= 2 // un-consume
            }
        }

        // single character tokens
        self.consume(1);
        match c {
            '(' => return Some(TokenType::LeftParen),
            ')' => return Some(TokenType::RightParen),
            '{' => return Some(TokenType::LeftCurly),
            '}' => return Some(TokenType::RightCurly),
            '+' => return Some(TokenType::Plus),
            '-' => return Some(TokenType::Minus),
            '*' => return Some(TokenType::Asterisk),
            '/' => return Some(TokenType::ForwardSlash),
            ';' => return Some(TokenType::Semicolon),
            ',' => return Some(TokenType::Comma),
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

        // strings
        if c == '"' {
            self.consume(1);
            while self.test(|c| c != '"' && c != '\n') { self.consume(1) }
            self.consume(1);
            let contents = self.lexeme()[1..self.lexeme().len()-1].to_string();
            return Some(TokenType::String(Rc::new(contents)));
        }

        // reserved words
        const RESERVED: [(&str, TokenType); 9] = [
            ("fn", TokenType::Fn),
            ("if", TokenType::If),
            ("else", TokenType::Else),
            ("true", TokenType::True),
            ("false", TokenType::False),
            ("while", TokenType::While),
            ("return", TokenType::Return),
            ("print", TokenType::Print),
            ("var", TokenType::Var),
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
            return Some(TokenType::Identifier(String::from(self.lexeme())))
        }

        // failed
        None
    }
}

#[derive(Debug, PartialEq, Clone)]
enum Expr {
    Fn { args: Vec<String>, body: Rc<Expr>, env: Option<Rc<RefCell<Environment>>> },
    FnCall { name: String, args: Vec<Rc<Expr>> },
    VarDecl(String, Rc<Expr>),
    VarAssign(String, Rc<Expr>),
    Var(String),
    Block(Rc<Expr>),
    IfThenElse { cond: Rc<Expr>, if_true: Rc<Expr>, if_false: Rc<Expr> },
    While { cond: Rc<Expr>, body: Rc<Expr> },
    PrefixUnary { op: TokenType, expr: Rc<Expr> },
    InfixBinary { left: Rc<Expr>, op: TokenType, right: Rc<Expr> },
    Integer(i64),
    String(Rc<String>),
    Bool(bool),
    Unit, // the expression "x;" resolves to unit (C++ void / Rust unit)
}

impl ToString for Expr {
    fn to_string(&self) -> String {
        match self {
            Expr::Fn { args: _, body: _, env: _ } => panic!("Cannot pretty-print a raw function"),
            Expr::FnCall { name, args } => {
                let mut s = String::new();
                s.push_str(&name);
                s.push('(');
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { s.push_str(", "); }
                    s.push_str(&arg.to_string());
                }
                s.push(')');
                s
            },
            Expr::VarDecl(name, expr) => match expr.as_ref() {
                Expr::Fn { args, body, env: _ } => {
                    let mut s = String::new();
                    s.push_str(&format!("fn {}(", name));
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 { s.push_str(", ") }
                        s.push_str(arg);
                    }
                    s.push_str(") {\n");
                    s.push_str(&body.to_string());
                    s.push_str("\n}\n");
                    s
                },
                _ => format!("var {} = {}", name, expr.to_string())
            },
            Expr::VarAssign(name, expr) => format!("{} = {}", name, expr.to_string()),
            Expr::Var(name) => name.clone(),
            Expr::Block(expr) => format!("{{ {} }}", expr.to_string()),
            Expr::IfThenElse { cond, if_true, if_false } => match **if_false {
                Expr::Unit => format!(" if ({}) {{ {} }} ", cond.to_string(), if_true.to_string()),
                _ => format!(" if ({}) {{ {} }} else {{ {} }} ", cond.to_string(), if_true.to_string(), if_false.to_string()),
            },
            Expr::While { cond, body } => format!(" while ({}) {{ {} }} ", cond.to_string(), body.to_string()),
            Expr::PrefixUnary { op, expr } => format!("{}{}", op.to_string(), expr.to_string()),
            Expr::InfixBinary { left, op, right } => match op {
                TokenType::Semicolon => format!("{} {} {}", left.to_string(), op.to_string(), right.to_string()),
                _ => format!("({} {} {})", left.to_string(), op.to_string(), right.to_string())
            },
            Expr::Integer(n) => n.to_string(),
            Expr::String(s) => format!("\"{}\"", s),
            Expr::Bool(b) => b.to_string(),
            Expr::Unit => String::from(""),
        }
    }
}

// Variable declarations in a lexical scope
#[derive(Debug, PartialEq, Clone)]
struct Environment {
    symbols: HashMap<String, Rc<Expr>>,
    parent: Option<Rc<RefCell<Environment>>>,
}

impl Environment {
    fn new() -> Self {
        Environment {
            symbols: HashMap::new(),
            parent: None,
        }
    }
    fn new_with_parent(parent: Rc<RefCell<Environment>>) -> Self {
        Environment {
            symbols: HashMap::new(),
            parent: Some(parent),
        }
    }
    fn get(&self, name: &String) -> Option<Rc<Expr>> {
        self.symbols.get(name).cloned()
            .or_else(|| self.parent.as_ref().and_then(|p| p.borrow().get(name)))
    }
    // define a new variable in the current environment, shadowing any in parent ones
    fn define(&mut self, name: String, expr: Rc<Expr>) {
        self.symbols.insert(name, expr);
    }
    // set/update an existing variable, choosing the most recent definition
    fn set(&mut self, name: String, expr: Rc<Expr>) {
        if self.symbols.contains_key(&name) {
            self.symbols.insert(name, expr);
        } else {
            if let Some(parent) = &self.parent {
                parent.borrow_mut().set(name, expr);
            }
        }
    }
}

struct Parser<'a> {
    stream: TokenStream<'a>,
    root: Result<Expr, BaseError>,
    _pos: usize,
}

impl<'a> Parser<'a> {
    fn new(stream: TokenStream<'a>) -> Self {
        let mut p = Self {
            stream,
            root: Err(BaseError::Unknown),
            _pos: 0,
        };
        // parse root as statement
        p.root = p.stmt();
        if p.peek() != Some(TokenType::EOF) {
            match p.peek() {
                Some(t) => println!("Unexpected tokens: {}", t.to_string()),
                None => println!("Unexpected end of file"),
            }
        }
        p
    }
    fn peek(&self) -> Option<TokenType> {
        Some(self.stream.tokens[self._pos..].iter().nth(0)?.token.clone())
    }
    fn consume(&mut self) -> Option<TokenType> {
        let t = self.peek();
        self._pos += 1;
        t
    }
    fn consume_type(&mut self, token_type: TokenType, message: &str) -> Result<(), BaseError> {
        if self.peek() == Some(token_type) {
            self.consume();
            Ok(())
        } else {
            Err(BaseError::Syntax(message.to_string()))
        }
    }
    // stmt ::= expr ( ; expr )*
    fn stmt(&mut self) -> Result<Expr, BaseError> {
        let mut expr = self.expr()?;
        while self.peek() == Some(TokenType::Semicolon) {
            let op = self.consume().unwrap();
            let right = Rc::new(self.expr()?);
            expr = Expr::InfixBinary {
                left: Rc::new(expr),
                op, right,
            }
        };
        Ok(expr)
    }
    // expr ::= { stmt } | if (binary_expr) expr | if (binary_expr) expr else expr | while (binary_expr) expr
    // | fn identifier ( ( identifier (, identifier)* )? ) { stmt }
    // | print expr | return expr | var identifier = expr | identifier = expr | binary_expr
    fn expr(&mut self) -> Result<Expr, BaseError> {
        if self.peek() == Some(TokenType::LeftCurly) {
            // { stmt }
            self.consume();
            let s = self.stmt()?;
            self.consume_type(TokenType::RightCurly, "Missing } at end of block")?;
            Ok(Expr::Block(Rc::new(s)))
        } else if self.peek() == Some(TokenType::If) {
            // if (binary_expr) expr | if (binary_expr) expr else expr
            self.consume();
            self.consume_type(TokenType::LeftParen, "Missing ( after 'if'")?;
            let cond = Rc::new(self.binary_expr()?);
            self.consume_type(TokenType::RightParen, "Missing ) after 'if' condition")?;
            let if_true = Rc::new(self.expr()?);
            let if_false = Rc::new(if self.peek() == Some(TokenType::Else) {
                // if (binary_expr) expr else expr
                self.consume();
                self.expr()?
            } else {
                // if (binary_expr) expr
                Expr::Unit
            });
            Ok(Expr::IfThenElse { cond, if_true, if_false })
        } else if self.peek() == Some(TokenType::While) {
            // while (binary_expr) expr
            self.consume();
            self.consume_type(TokenType::LeftParen, "Missing ( after 'while'")?;
            let cond = Rc::new(self.binary_expr()?);
            self.consume_type(TokenType::RightParen, "Missing ) after 'while' condition")?;
            let body = Rc::new(self.expr()?);
            Ok(Expr::While { cond, body })
        } else if self.peek() == Some(TokenType::Fn) {
            // fn identifier ( ( identifier (, identifier)* )? ) { stmt }
            self.consume();
            if let Some(TokenType::Identifier(name)) = self.consume() {
                self.consume_type(TokenType::LeftParen, "Missing ( after function name")?;
                let mut args = Vec::new();
                loop {
                    if let Some(TokenType::Identifier(arg)) = self.peek() {
                        self.consume();
                        args.push(arg);
                        if self.peek() != Some(TokenType::Comma) {
                            break;
                        }
                        self.consume();
                    } else {
                        break
                    }
                }
                self.consume_type(TokenType::RightParen, "Missing ) after function arguments")?;
                self.consume_type(TokenType::LeftCurly, "Missing { after arguments")?;
                let body = self.stmt()?;
                self.consume_type(TokenType::RightCurly, "Missing } after function body")?;
                Ok(Expr::VarDecl(
                    name,
                    Rc::new(Expr::Fn { args, body: Rc::new(body), env: None }),
                ))
            } else {
                Err(BaseError::Syntax("Missing function name".to_string()))
            }
        } else if self.peek() == Some(TokenType::Print) || self.peek() == Some(TokenType::Return) {
            // print expr | return expr
            let op = self.consume().unwrap();
            let expr = Rc::new(self.expr()?);
            Ok(Expr::PrefixUnary { op, expr })
        } else if self.peek() == Some(TokenType::Var) {
            // var identifier = expr
            self.consume();
            if let Some(TokenType::Identifier(ident)) = self.consume() {
                self.consume_type(TokenType::Equal, "Missing equal sign after variable name")?;
                let expr = Rc::new(self.expr()?);
                Ok(Expr::VarDecl(ident, expr))
            } else {
                Err(BaseError::Syntax(String::from("Missing identifier after 'var'")))
            }
        } else {
            // identifier = expr | binary_expr
            let val = self.binary_expr()?;
            match val {
                Expr::Var(name) => {
                    if self.peek() == Some(TokenType::Equal) {
                        // identifier = expr
                        self.consume();
                        let expr = Rc::new(self.expr()?);
                        Ok(Expr::VarAssign(name, expr))
                    } else {
                        Ok(Expr::Var(name))
                    }
                },
                _ => return Ok(val),
            }
        }
    }
    // handles all of the precedence levels of binary expressions recursively
    // an example of one level is binary_expr_60 ::= binary_expr_70 ( (+ | -) binary_expr_70 )*
    fn binary_expr(&mut self) -> Result<Expr, BaseError> {
        self.binary_expr_impl(1) // start from precedence 1
    }
    fn binary_expr_impl(&mut self, prec: u8) -> Result<Expr, BaseError> {
        // generic parser for infix binary left-associative operators
        let mut x = self.call()?;
        loop {
            let op = self.peek().unwrap();
            if op.fixity() < prec {
                return Ok(x)
            }
            self.consume();
            let y = self.binary_expr_impl(op.fixity() + 1)?;
            x = Expr::InfixBinary {
                left: Rc::new(x),
                op,
                right: Rc::new(y),
            }
        }
    }
    // call ::= identifier(expr (, expr)*) | unary
    fn call(&mut self) -> Result<Expr, BaseError> {
        // identifier(expr (, expr)*)
        if let Some(TokenType::Identifier(name)) = self.peek() {
            self.consume();
            if self.peek() == Some(TokenType::LeftParen) {
                self.consume();
                let mut args = Vec::new();
                while self.peek() != Some(TokenType::RightParen) {
                    args.push(Rc::new(self.expr()?));
                    if self.peek() != Some(TokenType::Comma) {
                        break;
                    }
                    self.consume();
                }
                self.consume_type(TokenType::RightParen, "Missing ) after function arguments")?;
                return Ok(Expr::FnCall { name, args })
            }
            self._pos -= 1 // undo consume
        }
        // unary
        self.unary()
    }
    // unary ::= ( "!" | "-" ) unary | primary
    fn unary(&mut self) -> Result<Expr, BaseError> {
        if self.peek() == Some(TokenType::Not) || self.peek() == Some(TokenType::Minus) {
            // recursive case
            let op = self.consume().unwrap();
            let expr = Rc::new(self.unary()?);
            Ok(Expr::PrefixUnary { op, expr })
        } else {
            // base case
            self.primary()
        }
    }
    // primary ::= NUMBER | STRING | true | false | IDENTIFIER | "(" expr ")"
    fn primary(&mut self) -> Result<Expr, BaseError> {
        match self.consume() {
            Some(TokenType::Natural(n)) => Ok(Expr::Integer(n as i64)),
            Some(TokenType::String(s)) => Ok(Expr::String(s)),
            Some(TokenType::True) => Ok(Expr::Bool(true)),
            Some(TokenType::False) => Ok(Expr::Bool(false)),
            Some(TokenType::Identifier(name)) => Ok(Expr::Var(name)),
            Some(TokenType::LeftParen) => {
                let expr = self.expr()?;
                match self.peek() {
                    Some(TokenType::RightParen) => {
                        self.consume();
                        Ok(expr)
                    },
                    // TODO: when we encounter a syntax error we should throw an exception
                    // and catch it in the statement parser to synchronise; drop all tokens until
                    // the next statement, and continue parsing to accumulate later errors
                    Some(tok) => Err(BaseError::Syntax(format!("Unexpected token: {}", tok.to_string()))),
                    None => Err(BaseError::Unknown),
                }
            },
            Some(tok) => Err(BaseError::Syntax(format!("Unexpected token: {}", tok.to_string()))),
            _ => Err(BaseError::Unknown),
        }
    }
}

struct Interpreter {
    global: Rc<RefCell<Environment>>,
}

impl Interpreter {
    fn new() -> Self {
        Self {
            global: Rc::new(RefCell::new(Environment::new())),
        }
    }
    fn interpret_program(&mut self, program: Rc<Expr>) -> Result<Rc<Expr>, BaseError> {
        self.interpret(program, self.global.clone())
    }
    fn interpret(&self, program: Rc<Expr>, env: Rc<RefCell<Environment>>) -> Result<Rc<Expr>, BaseError> {
        match program.as_ref() {
            Expr::VarDecl(name, expr) => {
                let value = self.interpret(expr.clone(), env.clone())?;
                env.borrow_mut().define(name.clone(), value.clone());
                Ok(value)
            },
            Expr::VarAssign(name, expr) => {
                let value = self.interpret(expr.clone(), env.clone())?;
                env.borrow_mut().set(name.clone(), value.clone());
                Ok(value)
            },
            Expr::Var(name) => {
                let value = env.borrow().get(&name);
                match value {
                    Some(value) => Ok(value),
                    None => Err(BaseError::Name(format!("Undefined variable {}", name))),
                }
            },
            Expr::Fn { args, body, env: _ } => {
                Ok(Rc::new(Expr::Fn {
                    args: args.clone(),
                    body: body.clone(),
                    env: Some(env.clone()),
                }))
            },
            Expr::FnCall { name, args } => {
                match env.borrow().get(&name) {
                    Some(f) => match f.as_ref() {
                        Expr::Fn { args: param_names, body, env: parent_scope } => {
                            if param_names.len() != args.len() {
                                return Err(BaseError::Syntax(format!("Function {} expects {} arguments", name, param_names.len())));
                            }
                            let scope = Rc::new(RefCell::new(Environment::new_with_parent(parent_scope.as_ref().unwrap().clone())));
                            for (name, arg) in param_names.iter().zip(args.iter()) {
                                scope.borrow_mut().define(name.clone(), self.interpret(arg.clone(), env.clone())?);
                            }
                            self.interpret(body.clone(), scope)
                        },
                        _ => Err(BaseError::Syntax(format!("Expected function, found {}", f.to_string()))),
                    },
                    None => Err(BaseError::Name(format!("Undefined function {}", name))),
                }
            },
            Expr::Block(expr) => {
                let scope = Rc::new(RefCell::new(Environment::new_with_parent(env.clone())));
                Ok(self.interpret(expr.clone(), scope)?)
            },
            Expr::IfThenElse { cond, if_true, if_false } => {
                let b = self.interpret(cond.clone(), env.clone())?;
                if self.cast_bool(b, env.clone())? {
                    self.interpret(if_true.clone(), env)
                } else {
                    self.interpret(if_false.clone(), env)
                }
            },
            Expr::While { cond, body } => {
                while self.cast_bool(self.interpret(cond.clone(), env.clone())?, env.clone())? {
                    self.interpret(body.clone(), env.clone())?;
                }
                Ok(Rc::new(Expr::Unit))  // TODO: make singleton unit
            },
            Expr::PrefixUnary { op, expr } => {
                let v = self.interpret(expr.clone(), env)?;
                match (op, v.as_ref()) {
                    (TokenType::Print, Expr::Integer(n)) => { println!("{}", n); Ok(Rc::new(Expr::Unit)) },  // TODO: make singleton unit
                    (TokenType::Print, Expr::String(s)) => { println!("{}", s); Ok(Rc::new(Expr::Unit)) },
                    (TokenType::Print, Expr::Bool(b)) => { println!("{}", b); Ok(Rc::new(Expr::Unit)) },
                    (TokenType::Print, Expr::Unit) => { println!("{}", Expr::Unit.to_string()); Ok(Rc::new(Expr::Unit)) },
                    (TokenType::Print, v) => Err(BaseError::Syntax(format!("Cannot print unexpected value: {}", v.to_string()))),

                    // exit the function early with the value
                    (TokenType::Return, _) => Ok(v.clone()),

                    (TokenType::Not, Expr::Bool(b)) => Ok(Rc::new(Expr::Bool(!b))),
                    (TokenType::Not, v) => Err(BaseError::Type(format!("Cannot use boolean negation on {}", v.to_string()))),

                    (TokenType::Minus, Expr::Integer(n)) => Ok(Rc::new(Expr::Integer(-n))),
                    (TokenType::Minus, v) => Err(BaseError::Type(format!("Cannot negate {}", v.to_string()))),

                    _ => Err(BaseError::Unknown)
                }
            },
            Expr::InfixBinary { left, op, right } => {
                let l = self.interpret(left.clone(), env.clone())?;
                let r = self.interpret(right.clone(), env)?;
                match (l.as_ref(), op, r.as_ref()) {
                    (_, TokenType::Semicolon, _) => Ok(r.clone()),

                    (Expr::Integer(x), TokenType::Plus, Expr::Integer(y)) => Ok(Rc::new(Expr::Integer(x + y))),
                    (Expr::String(x), TokenType::Plus, Expr::String(y)) => Ok(Rc::new(Expr::String(Rc::new(format!("{}{}", x, y))))),
                    (l, TokenType::Plus, r) => Err(BaseError::Type(format!("Cannot add {} and {}", l.to_string(), r.to_string()))),

                    (Expr::Integer(x), TokenType::Minus, Expr::Integer(y)) => Ok(Rc::new(Expr::Integer(x - y))),
                    (l, TokenType::Minus, r) => Err(BaseError::Type(format!("Cannot subtract {} and {}", l.to_string(), r.to_string()))),

                    (Expr::Integer(x), TokenType::Asterisk, Expr::Integer(y)) => Ok(Rc::new(Expr::Integer(x * y))),
                    (l, TokenType::Asterisk, r) => Err(BaseError::Type(format!("Cannot multiple {} and {}", l.to_string(), r.to_string()))),

                    (Expr::Integer(x), TokenType::ForwardSlash, Expr::Integer(y)) => Ok(Rc::new(Expr::Integer(x / y))),
                    (l, TokenType::ForwardSlash, r) => Err(BaseError::Type(format!("Cannot divide {} and {}", l.to_string(), r.to_string()))),

                    (_, TokenType::Comma, _) => Err(BaseError::NotImplemented),
                    (l, TokenType::Equal, r) => Err(BaseError::Syntax(format!("Equality operator used in invalid context: {} = {}", l.to_string(), r.to_string()))),

                    (x, TokenType::EqualEqual, y) => Ok(Rc::new(Expr::Bool(x == y))),
                    (x, TokenType::NotEqual, y) => Ok(Rc::new(Expr::Bool(x != y))),

                    (Expr::Integer(x), TokenType::Greater, Expr::Integer(y)) => Ok(Rc::new(Expr::Bool(x > y))),
                    (l, TokenType::Greater, r) => Err(BaseError::Type(format!("Cannot compare {} and {}", l.to_string(), r.to_string()))),

                    (Expr::Integer(x), TokenType::GreaterEqual, Expr::Integer(y)) => Ok(Rc::new(Expr::Bool(x >= y))),
                    (l, TokenType::GreaterEqual, r) => Err(BaseError::Type(format!("Cannot compare {} and {}", l.to_string(), r.to_string()))),

                    (Expr::Integer(x), TokenType::Less, Expr::Integer(y)) => Ok(Rc::new(Expr::Bool(x < y))),
                    (l, TokenType::Less, r) => Err(BaseError::Type(format!("Cannot compare {} and {}", l.to_string(), r.to_string()))),

                    (Expr::Integer(x), TokenType::LessEqual, Expr::Integer(y)) => Ok(Rc::new(Expr::Bool(x <= y))),
                    (l, TokenType::LessEqual, r) => Err(BaseError::Type(format!("Cannot compare {} and {}", l.to_string(), r.to_string()))),

                    (Expr::Bool(x), TokenType::And, Expr::Bool(y)) => Ok(Rc::new(Expr::Bool(*x && *y))),
                    (l, TokenType::And, r) => Err(BaseError::Type(format!("Cannot use logical and on {} and {}", l.to_string(), r.to_string()))),

                    (Expr::Bool(x), TokenType::Or, Expr::Bool(y)) => Ok(Rc::new(Expr::Bool(*x || *y))),
                    (l, TokenType::Or, r) => Err(BaseError::Type(format!("Cannot use logical or on {} and {}", l.to_string(), r.to_string()))),

                    _ => Err(BaseError::Unknown)
                }
            }
            // literals
            _ => Ok(program),
        }
    }
    fn cast_bool(&self, expr: Rc<Expr>, env: Rc<RefCell<Environment>>) -> Result<bool, BaseError> {
        let v = self.interpret(expr, env)?;
        match v.as_ref() {
            Expr::Bool(false) => Ok(false),
            Expr::Integer(0) => Ok(false),
            Expr::String(s) => Ok(s.len() > 0),
            _ => Ok(true)
        }
    }
}

fn repl() {
    let stdin = io::stdin();
    for _line in stdin.lock().lines() {
        let line = _line.unwrap();
        let stream = TokenStream::new(&line);
        if let Some(e) = stream.err.clone() {
            println!("{}\n", e.to_string())
        }
        let parser = Parser::new(stream);
        // dbg!(parser.stream.tokens);
        // dbg!(parser.root);
        match parser.root {
            Ok(tree) => {
                // println!("{}", tree.to_string());
                let mut interpreter = Interpreter::new();
                match interpreter.interpret_program(Rc::new(tree)) {
                    Ok(normalized) => println!("{}\n", normalized.to_string()),
                    Err(e) => println!("{}\n", e.to_string())
                }
            },
            Err(e) => println!("{}\n", e.to_string())
        }
    }
}

fn execute(source: String) {
    let stream = TokenStream::new(&source);
    if let Some(e) = stream.err.clone() {
        println!("{}\n", e.to_string())
    }
    let parser = Parser::new(stream);
    // dbg!(parser.stream.tokens);
    match parser.root {
        Ok(tree) => {
            // println!("{}", tree.to_string());
            // dbg!(tree.clone());
            let mut interpreter = Interpreter::new();
            match interpreter.interpret_program(Rc::new(tree)) {
                Ok(normalized) => println!("{}\n", normalized.to_string()),
                Err(e) => println!("{}\n", e.to_string())
            }
        },
        Err(e) => println!("{}\n", e.to_string())
    }
}

fn main() {
    let fname = std::env::args().nth(1);
    match fname {
        Some(file) => {
            let contents = std::fs::read_to_string(file.clone());
            match contents {
                Ok(contents) => execute(contents),
                Err(_) => println!("Could not find file {}", file)
            }
        },
        None => repl()
    }
}
