use crate::compiler::ast::repr::{
    Index, Item, ItemKind, Module, Path, PathKind, TaskItemKind, AST,
};
use crate::compiler::hir;
use crate::compiler::hir::Name;
use crate::compiler::info;
use crate::compiler::info::diags::Diagnostic;
use crate::compiler::info::files::{ByteIndex, FileId, Loc};
use crate::compiler::info::names::NameId;
use crate::compiler::info::Info;

use std::fmt::{Display, Formatter, Result};

pub(crate) struct ASTDebug<'a> {
    ast: &'a AST,
    info: &'a Info,
}

impl AST {
    pub(crate) fn debug<'a>(&'a self, info: &'a Info) -> ASTDebug<'a> {
        ASTDebug { ast: self, info }
    }
}

impl<'a> Display for ASTDebug<'a> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        writeln!(f, "AST Modules [")?;
        for (path, module) in &self.ast.modules {
            writeln!(
                f,
                r#"    "{}" => ["#,
                self.info.resolve_to_names(*path).join("::")
            )?;
            for item in &module.items {
                match &item.kind {
                    ItemKind::Alias(x)  => writeln!(f, r#"        "{}","#, self.info.names.resolve(x.name.id))?,
                    ItemKind::Enum(x)   => writeln!(f, r#"        "{}","#, self.info.names.resolve(x.name.id))?,
                    ItemKind::Fun(x)    => writeln!(f, r#"        "{}","#, self.info.names.resolve(x.name.id))?,
                    ItemKind::Extern(x) => writeln!(f, r#"        "{}","#, self.info.names.resolve(x.name.id))?,
                    ItemKind::Task(x)   => writeln!(f, r#"        "{}","#, self.info.names.resolve(x.name.id))?,
                    ItemKind::Use(x)    => writeln!(f, r#"        "{}","#, self.info.resolve_to_names(x.path.id).join("::"))?,
                    ItemKind::Err       => writeln!(f, r#"        <Error>,"#)?,
                }
            }

            writeln!(f, "    ],")?;
        }
        writeln!(f, "]")?;
        Ok(())
    }
}
