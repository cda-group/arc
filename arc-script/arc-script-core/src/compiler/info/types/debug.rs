use crate::compiler::hir;
use crate::compiler::hir::HIR;
use crate::compiler::info::types::TypeId;
use crate::compiler::info::Info;

use std::fmt::{Display, Formatter, Result};

/// A wrapper struct around `TypeId` which implements `Display`.
/// Will display debug information when printed.
pub(crate) struct TypeDebug<'a> {
    ty: &'a TypeId,
    info: &'a Info,
    hir: &'a HIR,
}

impl TypeId {
    /// Wraps `NameId` inside a `NameDebug` struct.
    pub(crate) const fn debug<'a>(&'a self, hir: &'a HIR, info: &'a Info) -> TypeDebug<'a> {
        TypeDebug {
            ty: self,
            info,
            hir,
        }
    }
}

impl<'a> Display for TypeDebug<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "{:?}: {}",
            self.ty,
            hir::pretty(self.ty, self.hir, self.info)
        )
    }
}
