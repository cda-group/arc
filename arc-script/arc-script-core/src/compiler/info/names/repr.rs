use crate::compiler::hir::Name;
use crate::compiler::info::files::Loc;
use derive_more::From;
use smartstring::{LazyCompact, SmartString};

/// An interner for interning `Name`s into `NameId`s, and resolving the other way around.
#[derive(Default, Debug)]
pub(crate) struct NameInterner {
    pub(crate) store: lasso::Rodeo,
    buf: String,
}

pub type NameBuf = str;

/// The product of interning a `Name`.
#[derive(Debug, Clone, Copy, From, Shrinkwrap, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NameId(lasso::Spur);

impl NameInterner {
    /// Interns a `Name` to a `NameId`.
    pub(crate) fn intern(&mut self, name: impl AsRef<NameBuf>) -> NameId {
        self.store.get_or_intern(name).into()
    }

    /// Resolves a `Name` to a `NameId`.
    pub(crate) fn resolve(&self, id: NameId) -> &NameBuf {
        self.store.resolve(&id)
    }

    /// Returns a uniquified version of `name`.
    pub(crate) fn fresh_with_base(&mut self, name: Name) -> Name {
        let buf = self.resolve(name.id).to_owned();
        let uid = self.uniquify(&buf);
        Name::new(uid, name.loc)
    }

    /// Generates and interns fresh new name.
    pub(crate) fn fresh(&mut self) -> Name {
        Name::new(self.uniquify("x"), None)
    }

    /// Generates and interns fresh new name which begins with `base`.
    /// NB: This is probably not ideal for performance since a new string needs
    /// to be allocated.
    fn uniquify(&mut self, base: impl AsRef<NameBuf>) -> NameId {
        self.buf.clear();
        self.buf.push_str(base.as_ref());
        loop {
            self.buf.push('_');
            for c in ('0'..='9').chain('A'..='Z').chain('a'..='z') {
                self.buf.push(c);
                if self.store.get(&self.buf).is_none() {
                    return self.store.get_or_intern(&self.buf).into();
                } else {
                    self.buf.pop();
                }
            }
        }
    }
}
