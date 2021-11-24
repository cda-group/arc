use crate::hir::repr::{Index, Item, ItemKind, Path, HIR};
use crate::info::Info;

use arc_script_compiler_shared::Shrinkwrap;

use std::fmt::{Display, Formatter, Result};

#[derive(Shrinkwrap)]
pub(crate) struct HIRDebug<'a> {
    hir: &'a HIR,
    #[shrinkwrap(main_field)]
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
            write!(f, r#"    {:<30} => "#, self.resolve_to_names(*path).join("::"))?;
            match &item.kind {
                ItemKind::TypeAlias(_)  => writeln!(f, "Alias,")?,
                ItemKind::Enum(_)       => writeln!(f, "Enum,")?,
                ItemKind::Fun(_)        => writeln!(f, "Fun,")?,
                ItemKind::ExternFun(_)  => writeln!(f, "ExternFun,")?,
                ItemKind::ExternType(_) => writeln!(f, "ExternType,")?,
                ItemKind::Task(_)       => writeln!(f, "Task,")?,
                ItemKind::Variant(_)    => writeln!(f, "Variant,")?,
            }
        }
        writeln!(f, "]")?;
        Ok(())
    }
}
