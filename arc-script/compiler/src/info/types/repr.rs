use crate::hir;
pub(crate) use crate::hir::Type;
pub(crate) use crate::hir::TypeId;

use arc_script_compiler_shared::Educe;
use arc_script_compiler_shared::Map;
use arc_script_compiler_shared::MapEntry;
use arc_script_compiler_shared::Shrinkwrap;
use arc_script_compiler_shared::VecMap;

use bitmaps::BitOps;
use bitmaps::*;
use ena::unify::InPlace;
use ena::unify::NoError;
use ena::unify::Snapshot;
use ena::unify::UnificationTable;
use ena::unify::UnifyKey;
use ena::unify::UnifyValue;

use std::cell::RefCell;

/// A data structure for interning and unifying types of the [`crate::repr::hir`].
///
/// NB: No interner is used for the [`crate::repr::ast`] since those types may
/// contain nominals whose paths are not yet resolved.
/// NB: Wrapped in a [`std::cell::RefCell`] since the backing store
/// [`ena::unify::UnificationTable`] needs mutable access for resolving
/// types in [`ena::unify::UnificationTable::probe_value`].
#[derive(Educe)]
#[educe(Debug)]
pub(crate) struct TypeInterner {
    pub(crate) store: RefCell<UnificationTable<InPlace<TypeId>>>,
    /// Snapshot which stores the initial state of the unification table. This is
    /// for technical reasons needed to get access to the types inside the table.
    #[educe(Debug(ignore))]
    pub(crate) snapshot: Snapshot<InPlace<TypeId>>,
}

impl Default for TypeInterner {
    fn default() -> Self {
        let mut store = UnificationTable::default();
        let initial_state = store.snapshot();
        Self {
            store: RefCell::from(store),
            snapshot: initial_state,
        }
    }
}

impl TypeInterner {
    /// Returns a fresh type variable which is unified with the given the `TypeKind`.
    pub(crate) fn intern(&mut self, kind: impl Into<hir::TypeKind>) -> Type {
        self.store.get_mut().new_key(kind.into()).into()
    }

    /// Returns the `TypeKind` which `t` is unified with.
    pub(crate) fn resolve(&self, t: impl Into<TypeId>) -> hir::TypeKind {
        self.store.borrow_mut().probe_value(t.into())
    }

    /// Returns a fresh type variable.
    pub(crate) fn fresh(&mut self) -> Type {
        self.store
            .get_mut()
            .new_key(hir::TypeKind::default())
            .into()
    }

    /// Returns the root of a type variable.
    pub(crate) fn root(&self, id: impl Into<TypeId>) -> TypeId {
        self.store.borrow_mut().find(id)
    }

    /// Collects all types in the unification-table into a map.
    pub(crate) fn collect(&mut self) -> Map<Type, hir::TypeKind> {
        let ts = self.store.get_mut().vars_since_snapshot(&self.snapshot);
        let mut map = Map::default();
        for i in ts.start.0..ts.end.0 {
            let t = Type::new(TypeId(i));
            match map.entry(t) {
                MapEntry::Occupied(_e) => {}
                MapEntry::Vacant(e) => {
                    e.insert(self.resolve(t));
                }
            }
        }
        map
    }
}

impl UnifyKey for TypeId {
    type Value = hir::TypeKind;

    fn index(&self) -> u32 {
        **self
    }

    fn from_index(id: u32) -> Self {
        Self(id)
    }

    fn tag() -> &'static str {
        "Type"
    }
}

impl UnifyValue for hir::TypeKind {
    type Error = NoError;

    /// Unifies two `TypeKind`s without the possibility of failing.
    fn unify_values(kind0: &Self, kind1: &Self) -> std::result::Result<Self, Self::Error> {
        use hir::TypeKind::*;
        match (&kind0, &kind1) {
            (Unknown(c0), Unknown(c1)) => Ok(Unknown(hir::Constraint(**c0 | **c1))),
            (Err, _) | (_, Err) => Ok(Err),
            (_, Unknown(_) | Err) => Ok(kind0.clone()),
            (Unknown(_) | Err, _) => Ok(kind1.clone()),
            _ => Ok(kind1.clone()),
        }
    }
}
