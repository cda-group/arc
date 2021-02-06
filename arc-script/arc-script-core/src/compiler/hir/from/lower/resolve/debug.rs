use crate::compiler::hir::from::lower::resolve::stack::SymbolStack;
use crate::compiler::hir::from::lower::resolve::table::SymbolTable;
use crate::compiler::hir::from::lower::resolve::Resolver;
use crate::compiler::info::Info;

use std::fmt::{Display, Formatter, Result};

pub(crate) struct ResolverDebug<'a> {
    res: &'a Resolver,
    info: &'a Info,
}
pub(crate) struct SymbolTableDebug<'a> {
    table: &'a SymbolTable,
    info: &'a Info,
}

pub(crate) struct SymbolStackDebug<'a> {
    stack: &'a SymbolStack,
    info: &'a Info,
}

pub(crate) struct Tab(usize);

impl Display for Tab {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        for i in 0..self.0 {
            write!(f, "    ")?;
        }
        Ok(())
    }
}

impl SymbolTable {
    pub(crate) fn debug<'a>(&'a self, info: &'a Info) -> SymbolTableDebug<'a> {
        SymbolTableDebug { table: self, info }
    }
}

impl SymbolStack {
    pub(crate) fn debug<'a>(&'a self, info: &'a Info) -> SymbolStackDebug<'a> {
        SymbolStackDebug { stack: self, info }
    }
}

impl Resolver {
    pub(crate) fn debug<'a>(&'a self, info: &'a Info) -> ResolverDebug<'a> {
        ResolverDebug { res: self, info }
    }
}

impl<'a> Display for SymbolTableDebug<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        writeln!(f, "SymbolTable: {{")?;
        writeln!(f, "    Imports: [")?;
        for (alias, target) in &self.table.imports {
            writeln!(
                f,
                "        {} => {}",
                self.info.resolve_to_names(*alias).join("::"),
                self.info.resolve_to_names(*target).join("::")
            )?;
        }
        writeln!(f, "    ],")?;
        writeln!(f, "    Declarations: [")?;
        for (path, decl) in &self.table.declarations {
            writeln!(
                f,
                "        {} => {:?}",
                self.info.resolve_to_names(*path).join("::"),
                decl
            )?;
        }
        writeln!(f, "    ],")?;
        writeln!(f, "    Compressed: [")?;
        for path in &self.table.compressed {
            writeln!(
                f,
                "        {}",
                self.info.resolve_to_names(*path).join("::"),
            )?;
        }
        writeln!(f, "    ]")?;
        writeln!(f, "}}")?;
        Ok(())
    }
}

fn indent(i: usize, f: &mut Formatter<'_>) -> Result {
    for x in 0..i {
        write!(f, "    ")?;
    }
    Ok(())
}

impl<'a> Display for SymbolStackDebug<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        writeln!(f, "SymbolStack: {{")?;
        for (i, frame) in self.stack.iter().enumerate() {
            writeln!(f, "{}Frame: {{", Tab(i + 1))?;
            for (j, scope) in frame.iter().enumerate() {
                writeln!(f, "{}Scope: {{", Tab(i + j + 2))?;
                for (name, unique_name) in scope.iter() {
                    writeln!(
                        f,
                        "{}{} => {}",
                        Tab(i + j + 3),
                        self.info.names.resolve(name.id),
                        self.info.names.resolve(unique_name.id),
                    )?;
                }
            }
            for (j, scope) in frame.iter().enumerate().rev() {
                writeln!(f, "{}}}", Tab(i + j + 2))?;
            }
        }
        for (i, frame) in self.stack.iter().enumerate().rev() {
            writeln!(f, "{}}}", Tab(i + 1))?;
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}

impl<'a> Display for ResolverDebug<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        writeln!(f, "{}", self.res.table.debug(self.info))?;
        writeln!(f, "{}", self.res.stack.debug(self.info))?;
        Ok(())
    }
}
