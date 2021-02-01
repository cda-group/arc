use crate::compiler::hir::Type;
use crate::compiler::info::types::TypeId;
use crate::compiler::info::TypeInterner;

pub(crate) trait Union<A, B> {
    /// Unions A with B
    fn union(&mut self, a: A, b: B);
}

impl<T: Into<Type>> Union<TypeId, T> for TypeInterner {
    fn union(&mut self, tv0: TypeId, ty1: T) {
        self.store.get_mut().union_value(tv0, ty1.into());
    }
}

impl Union<TypeId, TypeId> for TypeInterner {
    /// Unifies two type variables `a` and `b`.
    fn union(&mut self, a: TypeId, b: TypeId) {
        self.store.get_mut().union(a, b);
    }
}
