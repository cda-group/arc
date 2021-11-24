use crate::hir;
use crate::hir::Name;
use crate::info::types::Type;
use arc_script_compiler_shared::VecMap;

use super::equality::Equal;
use super::Context;

trait Subtype<A, B> {
    /// Checks if `a <: b`
    fn subtype(&mut self, a: A, b: B) -> bool;
}

impl Subtype<Vec<Type>, Vec<Type>> for Context<'_> {
    fn subtype(&mut self, ts0: Vec<Type>, ts1: Vec<Type>) -> bool {
        (ts0.len() == ts1.len())
            && ts0
                .into_iter()
                .zip(ts1.into_iter())
                .all(|(t0, t1)| self.subtype(t0, t1))
    }
}

impl Subtype<VecMap<Name, Type>, VecMap<Name, Type>> for Context<'_> {
    fn subtype(&mut self, fs0: VecMap<Name, Type>, fs1: VecMap<Name, Type>) -> bool {
        (fs0.len() >= fs1.len())
            && fs1
                .into_iter()
                .all(|(x1, t1)| matches!(fs0.get(&x1), Some(t0) if self.subtype(*t0, t1)))
    }
}

impl Subtype<Type, Type> for Context<'_> {
    /// Returns `true` if `A` is a subtype of `B`.
    #[rustfmt::skip]
    fn subtype(&mut self, t0: Type, t1: Type) -> bool {
        use hir::ScalarKind::*;
        use hir::TypeKind::*;
        let kind0 = self.types.resolve(t0);
        let kind1 = self.types.resolve(t1);
        match (kind0, kind1) {
            (Fun(ts0, t0), Fun(ts1, t1)) => self.subtype(ts1, ts0) && self.subtype(t0, t1),
            (Struct(fs0), Struct(fs1))   => self.subtype(fs0, fs1),
            _                            => self.eq(t0, t1),
        }
    }
}
