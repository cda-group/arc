use crate::compiler::info::names::NameId;
use crate::compiler::info::Info;

use std::fmt::{Display, Formatter, Result};

/// A wrapper struct around `NameId` which implements `Display`.
/// Will display debug information when printed.
pub(crate) struct NameDebug<'a> {
    name: &'a NameId,
    info: &'a Info,
}

impl NameId {
    /// Wraps `NameId` inside a `NameDebug` struct.
    pub(crate) const fn debug<'a>(&'a self, info: &'a Info) -> NameDebug<'a> {
        NameDebug { name: self, info }
    }
}

impl<'a> Display for NameDebug<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "{:?}: {}",
            self.name,
            self.info.names.resolve(*self.name)
        )
    }
}
