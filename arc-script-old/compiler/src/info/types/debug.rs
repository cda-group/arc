use crate::hir;
use crate::hir::HIR;
use crate::info::types::Type;
use crate::info::Info;

use std::fmt::{Display, Formatter, Result};

/// A wrapper struct around `Type` which implements `Display`.
/// Will display debug information when printed.
pub(crate) struct TypeDebug<'a> {
    ty: &'a Type,
    info: &'a Info,
    hir: &'a HIR,
}

impl Type {
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
        write!(f, "Type-{:<4?} => {}", self.ty.id.0, self.hir.pretty(self.ty, self.info))
    }
}
