use crate::info::names::NameId;
use crate::info::Info;
use lasso::Key;

use arc_script_compiler_shared::Shrinkwrap;

use std::fmt::{Display, Formatter, Result};

/// A wrapper struct around `NameId` which implements `Display`.
/// Will display debug information when printed.
#[derive(Shrinkwrap)]
pub(crate) struct NameDebug<'a> {
    name: &'a NameId,
    #[shrinkwrap(main_field)]
    pub(crate) info: &'a Info,
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
            "Name-{:<4?} => {}",
            self.name.into_usize(),
            self.names.resolve(self.name)
        )
    }
}
