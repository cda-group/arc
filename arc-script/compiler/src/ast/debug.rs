//! AST debugging utilities.

use crate::ast;
use crate::ast::repr::{Index, Item, ItemKind, Module, Path, TaskItemKind, AST};
use crate::info::Info;

use arc_script_compiler_shared::Shrinkwrap;

use std::fmt::{Display, Formatter, Result};

/// Wrapper around the AST which implements Display.
/// Can be printed to display AST-debug information.
#[derive(Shrinkwrap)]
pub(crate) struct ASTDebug<'a> {
    ast: &'a AST,
    #[shrinkwrap(main_field)]
    pub(crate) info: &'a Info,
}

impl AST {
    /// Wraps the AST inside an [`ASTDebug`].
    pub(crate) const fn debug<'a>(&'a self, info: &'a Info) -> ASTDebug<'a> {
        ASTDebug { ast: self, info }
    }
}

impl Display for ASTDebug<'_> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        writeln!(f, "AST Modules [")?;
        for (path, module) in &self.ast.modules {
            writeln!(f, "    {} => [", path.debug(self));
            for item in &module.items {
                write!(f, "        ")?;
                match &item.kind {
                    ItemKind::TypeAlias(x)  => write!(f, "{}", x.name.id.debug(self.info))?,
                    ItemKind::Enum(x)       => write!(f, "{}", x.name.id.debug(self.info))?,
                    ItemKind::Fun(x)        => write!(f, "{}", x.name.id.debug(self.info))?,
                    ItemKind::ExternFun(x)  => write!(f, "{}", x.decl.name.id.debug(self.info))?,
                    ItemKind::ExternType(x) => write!(f, "{}", x.name.id.debug(self.info))?,
                    ItemKind::Task(x)       => write!(f, "{}", x.name.id.debug(self.info))?,
                    ItemKind::Use(x)        => write!(f, "{}", x.path.id.debug(self.info))?,
                    ItemKind::Assign(x)     => write!(f, "{}", self.ast.pretty(&x.param.pat, self.info))?,
                    ItemKind::Err           => write!(f, "<Error>")?,
                }
                writeln!(f, ",")?;
            }
            writeln!(f, "    ],")?;
        }
        writeln!(f, "]")
    }
}
