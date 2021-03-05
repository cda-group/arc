use crate::compiler::hir::repr::{Index, Item, ItemKind, Path, HIR};

use crate::compiler::info::Info;

use std::fmt::{Display, Formatter, Result};

pub(crate) struct HIRDebug<'a> {
    hir: &'a HIR,
    info: &'a Info,
}

impl HIR {
    pub(crate) const fn debug<'a>(&'a self, info: &'a Info) -> HIRDebug<'a> {
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
                ItemKind::Alias(_x)   => writeln!(f, "alias,")?,
                ItemKind::Enum(_x)    => writeln!(f, "enum,")?,
                ItemKind::Fun(_x)     => writeln!(f, "fun,")?,
                ItemKind::Extern(_x)  => writeln!(f, "extern,")?,
                ItemKind::Task(_x)    => writeln!(f, "task,")?,
                ItemKind::Variant(_x) => writeln!(f, "variant,")?,
            }

        }
        writeln!(f, "]")?;
        Ok(())
    }
}
