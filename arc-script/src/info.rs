use crate::error::CompilerError;
use crate::symbols::SymbolTable;
use derive_more::Constructor;

#[derive(Constructor)]
pub struct Info<'i> {
    pub table: SymbolTable,
    pub errors: Vec<CompilerError>,
    pub source: &'i str,
}
