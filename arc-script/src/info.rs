use std::cell::RefCell;
use crate::error::CompilerError;
use crate::symbols::SymbolTable;
use crate::typer::Typer;
use derive_more::Constructor;

#[derive(Constructor)]
pub struct Info<'i> {
    pub table: SymbolTable,
    pub errors: Vec<CompilerError>,
    pub source: &'i str,
    pub typer: RefCell<Typer>,
}
