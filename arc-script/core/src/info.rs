use crate::prelude::*;
use std::cell::RefCell;

#[derive(Constructor)]
pub struct Info<'i> {
    pub table: SymbolTable,
    pub errors: Vec<CompilerError>,
    pub source: &'i str,
    pub typer: RefCell<Typer>,
}
