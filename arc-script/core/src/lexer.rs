#![allow(unused)]

use logos::{Lexer as LogosLexer, Logos};

pub const TABSIZE: usize = 4;

pub struct State {
    indents: usize,
}

impl Default for State {
    fn default() -> Self {
        let indents = 0;
        Self { indents }
    }
}

fn newline(lexer: &mut LogosLexer<LogosToken>) -> Option<Scope> {
    let offset = lexer
        .slice()
        .chars()
        .enumerate()
        .filter(|(_, c)| *c == '\n')
        .last()?
        .0;
    let len = lexer.span().len() + offset - 1;
    if len % TABSIZE == 0 {
        let old = lexer.extras.indents;
        lexer.extras.indents = len / TABSIZE;
        match lexer.extras.indents {
            new if new < old => Some(Scope::Dedent(old - new)),
            new if new == old + 1 => Some(Scope::Indent),
            new if new == old => Some(Scope::Unchanged),
            _ => None,
        }
    } else {
        None
    }
}

#[derive(Debug, PartialEq)]
pub enum Scope {
    Dedent(usize),
    Indent,
    Unchanged,
}

#[derive(Logos, Debug, PartialEq)]
#[logos(extras = State)]
pub enum LogosToken {
    #[regex(r"([\n\t\f] *)+", priority = 2, callback = newline)]
    Scope(Scope),

    #[token("if")]
    If,

    #[token("then")]
    Then,

    #[token("else")]
    Else,

    #[regex(r"[0-9]+", callback = |lex| lex.slice().parse())]
    I32(i32),

    #[regex(r"(true|false)", callback = |lex| lex.slice().parse())]
    Bool(bool),

    #[token(r":")]
    Colon,

    #[regex(r" +", priority = 1, callback = logos::skip)]
    #[error]
    Error,
}

pub struct Lexer<'i> {
    lexer: LogosLexer<'i, LogosToken>,
    dedents: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Token {
    Indent,
    Dedent,
    Error,
    If,
    Then,
    Else,
    I32(i32),
    Bool(bool),
    Colon,
}

impl<'i> Lexer<'i> {
    pub fn new(source: &'i str) -> Self {
        let lexer = LogosToken::lexer(source);
        let dedents = 0;
        Self { lexer, dedents }
    }
}

impl Iterator for Lexer<'_> {
    type Item = (usize, Token, usize);
    fn next(&mut self) -> Option<Self::Item> {
        let span = self.lexer.span();
        if self.dedents > 0 {
            self.dedents -= 1;
            return Some((span.start, Token::Dedent, span.end));
        }
        while let Some(token) = self.lexer.next() {
            let token = match token {
                LogosToken::Scope(Scope::Dedent(dedents)) => {
                    self.dedents = dedents - 1;
                    Token::Dedent
                }
                LogosToken::Scope(Scope::Indent) => Token::Indent,
                LogosToken::Scope(Scope::Unchanged) => continue,
                LogosToken::If => Token::If,
                LogosToken::Then => Token::Then,
                LogosToken::Else => Token::Else,
                LogosToken::I32(v) => Token::I32(v),
                LogosToken::Bool(v) => Token::Bool(v),
                LogosToken::Colon => Token::Colon,
                LogosToken::Error => Token::Error,
            };
            return Some((span.start, token, span.end));
        }
        if self.lexer.extras.indents > 0 {
            self.lexer.extras.indents -= 1;
            Some((span.start, Token::Dedent, span.end))
        } else {
            None
        }
    }
}

#[test]
fn test() {
    use indoc::indoc;
    let source = indoc! {"
        if true:
            if true:
                1
            else:
                2
                2
        else:
            if true:
                3
            else:
                4
    "};
    println!("{}", source);
    let mut lexer = Lexer::new(source);

    assert_eq!(lexer.next().unwrap().1, Token::If);
    assert_eq!(lexer.next().unwrap().1, Token::Bool(true));
    assert_eq!(lexer.next().unwrap().1, Token::Colon);
    assert_eq!(lexer.next().unwrap().1, Token::Indent);
    assert_eq!(lexer.next().unwrap().1, Token::If);
    assert_eq!(lexer.next().unwrap().1, Token::Bool(true));
    assert_eq!(lexer.next().unwrap().1, Token::Colon);
    assert_eq!(lexer.next().unwrap().1, Token::Indent);
    assert_eq!(lexer.next().unwrap().1, Token::I32(1));
    assert_eq!(lexer.next().unwrap().1, Token::Dedent);
    assert_eq!(lexer.next().unwrap().1, Token::Else);
    assert_eq!(lexer.next().unwrap().1, Token::Colon);
    assert_eq!(lexer.next().unwrap().1, Token::Indent);
    assert_eq!(lexer.next().unwrap().1, Token::I32(2));
    assert_eq!(lexer.next().unwrap().1, Token::I32(2));
    assert_eq!(lexer.next().unwrap().1, Token::Dedent);
    assert_eq!(lexer.next().unwrap().1, Token::Dedent);
    assert_eq!(lexer.next().unwrap().1, Token::Else);
    assert_eq!(lexer.next().unwrap().1, Token::Colon);
    assert_eq!(lexer.next().unwrap().1, Token::Indent);
    assert_eq!(lexer.next().unwrap().1, Token::If);
    assert_eq!(lexer.next().unwrap().1, Token::Bool(true));
    assert_eq!(lexer.next().unwrap().1, Token::Colon);
    assert_eq!(lexer.next().unwrap().1, Token::Indent);
    assert_eq!(lexer.next().unwrap().1, Token::I32(3));
    assert_eq!(lexer.next().unwrap().1, Token::Dedent);
    assert_eq!(lexer.next().unwrap().1, Token::Else);
    assert_eq!(lexer.next().unwrap().1, Token::Colon);
    assert_eq!(lexer.next().unwrap().1, Token::Indent);
    assert_eq!(lexer.next().unwrap().1, Token::I32(4));
    assert_eq!(lexer.next().unwrap().1, Token::Dedent);
    assert_eq!(lexer.next().unwrap().1, Token::Dedent);
    assert_eq!(lexer.next(), None);
}
