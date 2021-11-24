use crate::hir;


use arc_script_compiler_shared::VecMap;

use super::Context;

pub(crate) trait Equal<A, B> {
    /// Checks if `a == b`
    fn eq(&mut self, a: A, b: B) -> bool;
}

impl Equal<Vec<hir::Type>, Vec<hir::Type>> for Context<'_> {
    fn eq(&mut self, ts0: Vec<hir::Type>, ts1: Vec<hir::Type>) -> bool {
        (ts0.len() == ts1.len())
            && ts0
                .into_iter()
                .zip(ts1.into_iter())
                .all(|(t0, t1)| self.eq(t0, t1))
    }
}

impl Equal<VecMap<hir::Name, hir::Type>, VecMap<hir::Name, hir::Type>> for Context<'_> {
    fn eq(&mut self, fs0: VecMap<hir::Name, hir::Type>, fs1: VecMap<hir::Name, hir::Type>) -> bool {
        (fs0.len() == fs1.len())
            && fs0
                .into_iter()
                .all(|(x0, t0)| matches!(fs1.get(&x0), Some(t1) if self.eq(t0, *t1)))
    }
}

impl Equal<hir::Type, hir::Type> for Context<'_> {
    #[rustfmt::skip]
    fn eq(&mut self, t0: hir::Type, t1: hir::Type) -> bool {
        use hir::TypeKind::*;
        let kind0 = self.types.resolve(t0);
        let kind1 = self.types.resolve(t1);
        match (kind0, kind1) {
            (Array(t0, _s0), Array(t1, _s1)) => self.eq(t0, t1),
            (Fun(ts0, t0), Fun(ts1, t1))     => self.eq(ts0, ts1) && self.eq(t0,t1),
            (Nominal(x0), Nominal(x1))       => *x0 == *x1,
            (Scalar(kind0), Scalar(kind1))   => kind0 == kind1,
            (Stream(t0), Stream(t1))         => self.eq(t0, t1),
            (Struct(ft0), Struct(ft1))       => self.eq(ft0, ft1),
            (Tuple(ts0), Tuple(ts1))         => self.eq(ts0, ts1),
            (Unknown(_), Unknown(_))         => true,
            (Err, Err)                       => true,
            _                                => false
        }
    }
}
