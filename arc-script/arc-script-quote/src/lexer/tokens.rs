use proc_macro::{Delimiter, Ident, Literal};
use std::fmt::{self, Display};

#[derive(Clone, Debug)]
pub enum Token {
    Open(Delimiter),
    Close(Delimiter),
    Punct(char),
    Joint,
    Ident(Ident),
    Literal(Literal),
}
