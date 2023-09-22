#![allow(unused)]
pub mod tokens;

use std::convert::TryInto;

use diagnostics::Diagnostics;
use diagnostics::Error;
use info::ByteIndex;
use info::Info;
use regex::Regex;
use tokens::Token;

pub struct Lexer<'a> {
    id: usize,
    inner: logos::Lexer<'a, Token>,
    diagnostics: Diagnostics,
    regex: Regex,
}

impl<'a> Lexer<'a> {
    pub fn new(id: usize, source: &'a str) -> Self {
        Self {
            id,
            inner: logos::Lexer::new(source),
            diagnostics: Diagnostics::default(),
            regex: Regex::new("(?ms)^(?<lang>[a-z]+)(?<code>.*?)---").unwrap(),
        }
    }

    pub fn diagnostics(self) -> Diagnostics {
        self.diagnostics
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = (ByteIndex, Token, ByteIndex);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let token = self.inner.next()?;
            let span = self.inner.span();
            let start = span.start.try_into().unwrap();
            let end = span.end.try_into().unwrap();
            match token {
                Ok(Token::MinusMinusMinus) => {
                    let span = self.inner.span();
                    let beginning = span.end;
                    let m = self.regex.captures(&self.inner.remainder()).unwrap();
                    let lang = m["lang"].to_owned();
                    let code = m["code"].to_owned();
                    let len = m[0].len();
                    let end = beginning + len;
                    self.inner.bump(len);
                    return Some((start, Token::Inject((lang, code)), end.try_into().unwrap()));
                }
                Ok(token) => return Some((start, token, end)),
                Err(err) => {
                    let info = Info::new(self.id, start, end);
                    self.diagnostics
                        .push_error(Error::LexerInvalidToken { info, err })
                }
            }
        }
    }
}
