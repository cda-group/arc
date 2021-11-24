//! Converts rust tokens into arc-script tokens.
use crate::tokens;
use arc_script::ast::from::lexer::Token;
use proc_macro::Delimiter;

impl From<tokens::Token> for Token {
    fn from(token: tokens::Token) -> Self {
        match token {
            tokens::Token::Open(d) => match d {
                Delimiter::Parenthesis => Token::ParenL,
                Delimiter::Brace => Token::BraceL,
                Delimiter::Bracket => Token::BrackL,
                Delimiter::None => todo!()
            },
            tokens::Token::Close(d) => match d {
                Delimiter::Parenthesis => Token::ParenR,
                Delimiter::Brace => Token::BraceR,
                Delimiter::Bracket => Token::BrackR,
                Delimiter::None => todo!(),
            },
            tokens::Token::Punct(_) => {}
            tokens::Token::Joint => {}
            tokens::Token::Ident(_) => {}
            tokens::Token::Literal(_) => {}
        }
    }
}
