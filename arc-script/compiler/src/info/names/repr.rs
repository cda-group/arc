use crate::info::files::Loc;
use crate::info::files::Spanned;

use arc_script_compiler_macros::Loc;
use arc_script_compiler_macros::GetId;
use arc_script_compiler_shared::From;
use arc_script_compiler_shared::Hasher;
use arc_script_compiler_shared::Shrinkwrap;
use arc_script_compiler_shared::New;
use arc_script_compiler_shared::Educe;

use lasso::MiniSpur;
use lasso::Rodeo;

/// A key which can represent 2^16 names.
pub(crate) type Key = MiniSpur;
/// A store for interning names.
pub(crate) type Store = Rodeo<Key, Hasher>;

/// An identifier.
#[derive(Debug, Clone, Copy, Loc, Educe, New, GetId)]
#[educe(PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Name {
    /// Name identifier.
    pub id: NameId,
    /// Name location.
    #[educe(PartialEq(ignore), Eq(ignore), Hash(ignore))]
    #[educe(PartialOrd(ignore), Ord(ignore))]
    pub loc: Loc,
}

/// An interner for interning `Name`s into `NameId`s, and resolving the other way around.
#[derive(Debug)]
pub(crate) struct NameInterner {
    /// Where all names are stored.
    pub(crate) store: Store,
    /// Commonly occuring names.
    pub(crate) common: Common,
    /// Buffer used when generating new names.
    buf: String,
}

macro_rules! common {
    {
        $($name:ident:$literal:literal),* $(,)?
    } => {
        #[derive(Debug)]
        pub(crate) struct Common {
            $(pub(crate) $name: Name,)*
        }
        impl Common {
            fn new(store: &mut Store) -> Self {
                Self {
                    $($name: NameId(store.get_or_intern_static($literal)).into(),)*
                }
            }
        }
    }
}

/// Commonly occurring names.
common! {
    root: "crate",
    iinterface: "IInterface",
    ointerface: "OInterface",
    dummy: "__",
    val: "value",
    key: "key",
    on_event: "on_event",
    on_start: "on_start",
}

impl Default for NameInterner {
    fn default() -> Self {
        let mut store = Store::with_hasher(Hasher::default());
        let buf = String::new();
        let common = Common::new(&mut store);
        Self { store, common, buf }
    }
}

/// A kind of name.
pub(crate) type NameKind = str;

/// The product of interning a `Name`.
#[derive(Debug, Clone, Copy, From, Shrinkwrap, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NameId(Key);

impl NameInterner {
    /// Interns a `NameBuf` to a `NameId`.
    pub(crate) fn intern(&mut self, name: impl AsRef<NameKind>) -> NameId {
        self.store.get_or_intern(name).into()
    }

    /// Resolves a `NameId` to a `NameBuf`.
    pub(crate) fn resolve(&self, id: impl Into<NameId>) -> &NameKind {
        self.store.resolve(&id.into())
    }

    /// Returns a uniquified version of `name`.
    pub(crate) fn fresh_with_base(&mut self, name: Name) -> Name {
        let buf = self.resolve(name).to_owned();
        let uid = self.uniquify(&buf);
        Name::new(uid, name.loc)
    }

    /// Generates and interns fresh new name.
    pub(crate) fn fresh(&mut self) -> Name {
        Name::new(self.uniquify("x"), Loc::Fake)
    }

    /// Generates and interns fresh new name which begins with `base`.
    /// NB: This is probably not ideal for performance since a new string needs
    /// to be allocated.
    fn uniquify(&mut self, base: impl AsRef<NameKind>) -> NameId {
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
