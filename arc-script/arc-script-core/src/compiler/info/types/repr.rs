use crate::compiler::hir::{Type, TypeKind};
use educe::Educe;
use ena::unify::{
    InPlace, NoError, Snapshot, UnificationStoreMut, UnificationTable, UnifyKey, UnifyValue,
};

use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::collections::HashMap;

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

/// A type variable which maps to a [`crate::repr::hir::Type`].
#[derive(Shrinkwrap, Default, Debug, Eq, PartialEq, Copy, Clone, Hash)]
pub struct TypeId(pub(crate) u32);

impl TypeInterner {
    /// Returns a fresh type variable which is unified with the given type `ty`.
    pub(crate) fn intern(&mut self, ty: impl Into<Type>) -> TypeId {
        self.store.get_mut().new_key(ty.into())
    }

    /// Returns the type which `tv` is unified with.
    pub(crate) fn resolve(&self, tv: TypeId) -> Type {
        self.store.borrow_mut().probe_value(tv)
    }

    /// Returns a fresh type variable.
    pub(crate) fn fresh(&mut self) -> TypeId {
        self.store.get_mut().new_key(Type::default())
    }

    /// Returns the root of a type variable.
    pub(crate) fn root(&self, tv: TypeId) -> TypeId {
        self.store.borrow_mut().find(tv)
    }

    /// Collects all types in the unification-table into a hashmap.
    pub(crate) fn collect(&mut self) -> HashMap<TypeId, Type> {
        let tvs = self.store.get_mut().vars_since_snapshot(&self.snapshot);
        let mut map = HashMap::new();
        for i in tvs.start.0..tvs.end.0 {
            let tv = TypeId(i);
            match map.entry(tv) {
                Entry::Occupied(e) => {}
                Entry::Vacant(e) => {
                    let ty = self.resolve(tv);
                    e.insert(ty);
                }
            }
        }
        map
    }
}

impl UnifyKey for TypeId {
    type Value = Type;

    fn index(&self) -> u32 {
        **self
    }

    fn from_index(id: u32) -> TypeId {
        TypeId(id)
    }

    fn tag() -> &'static str {
        "Type"
    }
}

impl UnifyValue for Type {
    type Error = NoError;

    /// Unifies two type variables without the possibility of failing.
    /// TODO: Do not clone, instead use Rc<Type>
    fn unify_values(ty1: &Self, ty2: &Self) -> std::result::Result<Self, Self::Error> {
        use TypeKind::*;
        match (&ty1.kind, &ty2.kind) {
            (Unknown, _) | (Err, _) => Ok(ty2.clone()),
            (_, Unknown) | (_, Err) => Ok(ty1.clone()),
            _ => Ok(ty1.clone()),
        }
    }
}
