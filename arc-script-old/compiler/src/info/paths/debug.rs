use crate::info::paths::PathId;
use crate::info::Info;

use arc_script_compiler_shared::Shrinkwrap;

use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result;

/// A wrapper struct around `PathId` which implements `Display`.
/// Will print debug information when displayed.
#[derive(Shrinkwrap)]
pub(crate) struct PathDebug<'a> {
    path: &'a PathId,
    #[shrinkwrap(main_field)]
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
        let i: usize = (*self.path).into();
        let p = self.resolve_to_names(self.path).join("::");
        write!(f, "Path-{:<4?} => {}", i, p)
    }
}
