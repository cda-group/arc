use crate::hir::lower::ast::resolve::stack::SymbolStack;
use crate::hir::lower::ast::resolve::table::SymbolTable;
use crate::hir::lower::ast::resolve::Resolver;
use crate::info::Info;

use arc_script_compiler_shared::Shrinkwrap;

use std::fmt::{Display, Formatter, Result};

#[derive(Shrinkwrap)]
pub(crate) struct ResolverDebug<'a> {
    res: &'a Resolver,
    #[shrinkwrap(main_field)]
    info: &'a Info,
}

#[derive(Shrinkwrap)]
pub(crate) struct SymbolTableDebug<'a> {
    table: &'a SymbolTable,
    #[shrinkwrap(main_field)]
    info: &'a Info,
}

#[derive(Shrinkwrap)]
pub(crate) struct SymbolStackDebug<'a> {
    stack: &'a SymbolStack,
    #[shrinkwrap(main_field)]
    info: &'a Info,
}

pub(crate) struct Tab(usize);

impl Display for Tab {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        for _i in 0..self.0 {
            write!(f, "    ")?;
        }
        Ok(())
    }
}

impl SymbolTable {
    pub(crate) const fn debug<'a>(&'a self, info: &'a Info) -> SymbolTableDebug<'a> {
        SymbolTableDebug { table: self, info }
    }
}

impl SymbolStack {
    pub(crate) const fn debug<'a>(&'a self, info: &'a Info) -> SymbolStackDebug<'a> {
        SymbolStackDebug { stack: self, info }
    }
}

impl Resolver {
    pub(crate) const fn debug<'a>(&'a self, info: &'a Info) -> ResolverDebug<'a> {
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
                self.resolve_to_names(*alias).join("::"),
                self.resolve_to_names(*target).join("::")
            )?;
        }
        writeln!(f, "    ],")?;
        writeln!(f, "    Declarations: [")?;
        for (path, decl) in &self.table.declarations {
            writeln!(
                f,
                "        {:<14} => {:?}",
                self.resolve_to_names(*path).join("::"),
                decl
            )?;
        }
        writeln!(f, "    ],")?;
        writeln!(f, "    Compressed: [")?;
        for path in &self.table.compressed {
            writeln!(f, "        {}", self.resolve_to_names(*path).join("::"))?;
        }
        writeln!(f, "    ]")?;
        writeln!(f, "}}")?;
        Ok(())
    }
}

fn indent(i: usize, f: &mut Formatter<'_>) -> Result {
    for _x in 0..i {
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
                for (name, (unique_name, kind)) in scope.iter() {
                    writeln!(
                        f,
                        "{}{:<10} => {} ({:?})",
                        Tab(i + j + 3),
                        self.names.resolve(name),
                        self.names.resolve(unique_name),
                        kind,
                    )?;
                }
            }
            for (j, _scope) in frame.iter().enumerate().rev() {
                writeln!(f, "{}}}", Tab(i + j + 2))?;
            }
        }
        for (i, _frame) in self.stack.iter().enumerate().rev() {
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
