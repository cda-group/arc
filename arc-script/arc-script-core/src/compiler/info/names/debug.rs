use crate::compiler::info::names::NameId;
use crate::compiler::info::Info;

use std::fmt::{Display, Formatter, Result};

pub(crate) struct NameDebug<'a> {
    name: &'a NameId,
    info: &'a Info,
}

impl NameId {
    fn debug<'a>(&'a self, info: &'a Info) -> NameDebug<'a> {
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
