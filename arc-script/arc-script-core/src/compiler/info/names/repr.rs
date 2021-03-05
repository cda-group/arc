use crate::compiler::hir::Name;

use arc_script_core_shared::From;
use arc_script_core_shared::Hasher;
use arc_script_core_shared::Shrinkwrap;

use lasso::MiniSpur;
use lasso::Rodeo;

/// A key which can represent 2^16 names.
pub(crate) type Key = MiniSpur;
/// A store for interning names.
pub(crate) type Store = Rodeo<Key, Hasher>;

/// An interner for interning `Name`s into `NameId`s, and resolving the other way around.
#[derive(Debug)]
pub(crate) struct NameInterner {
    pub(crate) store: Store,
    pub(crate) common: Common,
    buf: String,
}

/// Commonly occurring names.
#[derive(Debug)]
pub(crate) struct Common {
    pub(crate) root: NameId,
    pub(crate) value: NameId,
}

impl Default for NameInterner {
    fn default() -> Self {
        let mut store = Store::with_hasher(Hasher::default());
        let buf = String::new();
        let common = Common {
            root: NameId(store.get_or_intern("crate")),
            value: NameId(store.get_or_intern("value")),
        };
        Self { store, common, buf }
    }
}

pub(crate) type NameBuf = str;

/// The product of interning a `Name`.
#[derive(Debug, Clone, Copy, From, Shrinkwrap, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NameId(Key);

impl NameInterner {
    /// Interns a `NameBuf` to a `NameId`.
    pub(crate) fn intern(&mut self, name: impl AsRef<NameBuf>) -> NameId {
        self.store.get_or_intern(name).into()
    }

    /// Resolves a `NameId` to a `NameBuf`.
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
