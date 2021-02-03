use crate::compiler::hir::from::lower::resolve::stack::SymbolStack;
use crate::compiler::hir::from::lower::resolve::table::SymbolTable;
use crate::compiler::hir::from::lower::resolve::Resolver;
use crate::compiler::info::Info;

use std::fmt::{Display, Formatter, Result};

pub(crate) struct SymbolTableDisplay<'a> {
    table: &'a SymbolTable,
    info: &'a Info,
}

impl SymbolTable {
    pub(crate) fn display<'a>(&'a self, info: &'a Info) -> SymbolTableDisplay<'a> {
        SymbolTableDisplay { table: self, info }
    }
}

impl<'a> Display for SymbolTableDisplay<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        writeln!(f);
        writeln!(f, "Imports: [");
        for (alias, target) in &self.table.imports {
            writeln!(
                f,
                "    {} => {}",
                self.info.resolve_to_names(*alias).join("::"),
                self.info.resolve_to_names(*target).join("::")
            );
        }
        writeln!(f, "]");
        writeln!(f, "Declarations: [");
        for (path, decl) in &self.table.declarations {
            writeln!(
                f,
                "    {} => {:?}",
                self.info.resolve_to_names(*path).join("::"),
                decl
            );
        }
        writeln!(f, "]");
        writeln!(f, "Compressed: [");
        for path in &self.table.compressed {
            writeln!(f, "    {}", self.info.resolve_to_names(*path).join("::"),);
        }
        writeln!(f, "]");
        Ok(())
    }
}
