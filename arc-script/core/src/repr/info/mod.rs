pub(crate) mod error;
pub(crate) mod opt;
pub(crate) mod symbols;
pub(crate) mod connector;

use crate::prelude::*;
use std::cell::RefCell;

#[derive(Constructor)]
pub struct Info<'i> {
    pub table: SymbolTable,
    pub errors: Vec<CompilerError>,
    pub source: &'i str,
    pub typer: RefCell<Typer>,
    pub opt: &'i Opt,
}
