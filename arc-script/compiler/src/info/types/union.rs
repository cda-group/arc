use crate::hir::Type;
use crate::hir::TypeKind;
use crate::info::types::TypeInterner;

pub(crate) trait Union<A, B> {
    /// Unions A with B
    fn union(&mut self, a: A, b: B);
}

impl<T: Into<TypeKind>> Union<Type, T> for TypeInterner {
    fn union(&mut self, t0: Type, t1: T) {
        self.store.get_mut().union_value(t0.id, t1.into());
    }
}

impl Union<Type, Type> for TypeInterner {
    /// Unifies two type variables `a` and `b`.
    fn union(&mut self, a: Type, b: Type) {
        self.store.get_mut().union(a.id, b.id);
    }
}
