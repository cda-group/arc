use crate::compiler::info::paths::PathId;
use crate::compiler::info::Info;

use std::fmt::{Display, Formatter, Result};

/// A wrapper struct around `PathId` which implements `Display`.
/// Will print debug information when displayed.
pub(crate) struct PathDebug<'a> {
    path: &'a PathId,
    info: &'a Info,
}

impl PathId {
    /// Wraps `PathId` inside a `PathDebug` struct.
    pub(crate) const fn debug<'a>(&'a self, info: &'a Info) -> PathDebug<'a> {
        PathDebug { path: self, info }
    }
}

impl<'a> Display for PathDebug<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "{:?}: {}",
            self.path,
            self.info.resolve_to_names(*self.path).join("::")
        )
    }
}
