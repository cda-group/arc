use crate::compiler::hir;
use crate::compiler::hir::repr::{Index, Item, ItemKind, Path, HIR};
use crate::compiler::hir::Name;
use crate::compiler::info;
use crate::compiler::info::diags::Diagnostic;
use crate::compiler::info::files::{ByteIndex, FileId, Loc};
use crate::compiler::info::names::NameId;
use crate::compiler::info::Info;

use std::fmt::{Display, Formatter, Result};

pub(crate) struct HIRDebug<'a> {
    hir: &'a HIR,
    info: &'a Info,
}

impl HIR {
    pub(crate) fn debug<'a>(&'a self, info: &'a Info) -> HIRDebug<'a> {
        HIRDebug { hir: self, info }
    }
}

impl<'a> Display for HIRDebug<'a> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        writeln!(f, "HIR Items [")?;
        for (path, item) in &self.hir.defs {
            write!(
                f,
                r#"    "{}" => "#,
                self.info.resolve_to_names(path.id).join("::")
            )?;
            match &item.kind {
                ItemKind::Alias(x)   => writeln!(f, "alias,")?,
                ItemKind::Enum(x)    => writeln!(f, "enum,")?,
                ItemKind::Fun(x)     => writeln!(f, "fun,")?,
                ItemKind::Extern(x)  => writeln!(f, "extern,")?,
                ItemKind::Task(x)    => writeln!(f, "task,")?,
                ItemKind::State(x)   => writeln!(f, "state,")?,
                ItemKind::Variant(x) => writeln!(f, "variant,")?,
            }

        }
        writeln!(f, "]")?;
        Ok(())
    }
}
