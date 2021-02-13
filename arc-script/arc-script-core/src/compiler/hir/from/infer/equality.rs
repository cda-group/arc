use crate::compiler::hir::{
    self, BinOpKind, Dim, DimKind, Expr, ExprKind, Fun, LitKind, Name, Param, ScalarKind, Shape,
    Type, TypeKind, UnOpKind, HIR,
};

use crate::compiler::info::types::TypeId;

use crate::compiler::shared::VecMap;

use super::Context;

pub(crate) trait Equal<A, B> {
    /// Checks if `a == b`
    fn eq(&mut self, a: A, b: B) -> bool;
}

impl Equal<Vec<TypeId>, Vec<TypeId>> for Context<'_> {
    fn eq(&mut self, ts0: Vec<TypeId>, ts1: Vec<TypeId>) -> bool {
        (ts0.len() == ts1.len())
            && ts0
                .into_iter()
                .zip(ts1.into_iter())
                .all(|(t0, t1)| self.eq(t0, t1))
    }
}

impl Equal<VecMap<Name, TypeId>, VecMap<Name, TypeId>> for Context<'_> {
    fn eq(&mut self, fs0: VecMap<Name, TypeId>, fs1: VecMap<Name, TypeId>) -> bool {
        (fs0.len() == fs1.len())
            && fs0
                .into_iter()
                .all(|(x0, t0)| matches!(fs1.get(&x0), Some(t1) if self.eq(t0, *t1)))
    }
}

impl Equal<TypeId, TypeId> for Context<'_> {
    #[rustfmt::skip]
    fn eq(&mut self, t0: TypeId, t1: TypeId) -> bool {
        use TypeKind::*;
        let ty0 = self.info.types.resolve(t0);
        let ty1 = self.info.types.resolve(t1);
        match (ty0.kind, ty1.kind) {
            (Array(t0, _s0), Array(t1, _s1)) => self.eq(t0, t1),
            (Fun(ts0, t0), Fun(ts1, t1))     => self.eq(ts0, ts1) && self.eq(t0,t1),
            (Map(t00, t01), Map(t10, t11))   => self.eq(t00, t10) && self.eq(t01, t11),
            (Nominal(x0), Nominal(x1))       => x0 == x1,
            (Optional(t0), Optional(t1))     => self.eq(t0, t1),
            (Scalar(kind0), Scalar(kind1))   => kind0 == kind1,
            (Set(t0), Set(t1))               => self.eq(t0, t1),
            (Stream(t0), Stream(t1))         => self.eq(t0, t1),
            (Struct(ft0), Struct(ft1))       => self.eq(ft0, ft1),
            (Tuple(ts0), Tuple(ts1))         => self.eq(ts0, ts1),
            (Vector(t0), Vector(t1))         => self.eq(t0, t1),
            (Unknown, Unknown)               => true,
            (Err, Err)                       => true,
            _                                => false
        }
    }
}
