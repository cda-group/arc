use crate::hir;
use crate::info::diags::Diagnostic;
use crate::info::diags::Diagnostic::Warning;
use crate::info::diags::Error;
use crate::info::files::Loc;
use crate::info::types::union::Union;
use crate::info::types::TypeId;

use super::Context;

pub(crate) trait Unify<A, B> {
    /// Unifies A with B
    fn unify(&mut self, a: A, b: B);
}

impl<T: Into<hir::TypeKind>, const C: usize> Unify<hir::Type, [T; C]> for Context<'_> {
    fn unify(&mut self, t0: hir::Type, kinds: [T; C]) {
        for kind in kinds {
            self.unify(t0, kind)
        }
    }
}

/// Unifies a type variable `a` with a type `b`.
impl<T: Into<hir::TypeKind>> Unify<hir::Type, T> for Context<'_> {
    fn unify(&mut self, t0: hir::Type, kind: T) {
        let t1: hir::Type = self.types.intern(kind.into());
        self.unify(t0, t1);
    }
}

impl<const C: usize> Unify<hir::Type, [hir::Type; C]> for Context<'_> {
    fn unify(&mut self, t0: hir::Type, ts: [hir::Type; C]) {
        for t1 in ts {
            self.unify(t0, t1)
        }
    }
}

/// Unifies two polytypes, i.e., types which may contain type variables
impl Unify<hir::Type, hir::Type> for Context<'_> {
    fn unify(&mut self, t0: hir::Type, t1: hir::Type) {
        use hir::TypeKind::*;
        tracing::trace!(
            "unify({}, {})",
            self.hir.pretty(&t0, self.info),
            self.hir.pretty(&t1, self.info),
        );
        let kind0 = self.types.resolve(t0);
        let kind1 = self.types.resolve(t1);
        match (&kind0, &kind1) {
            // Unify Unknown types
            (Unknown(_), Unknown(_)) => self.types.union(t0, t1),
            (Unknown(_), _) => self.types.union(t0, kind1),
            (_, Unknown(_)) => self.types.union(t1, kind0),
            // Zip other types and unify their inner Unknown types
            (Nominal(x0), Nominal(x1)) if x0.id == x1.id => {}
            (Scalar(kind0), Scalar(kind1)) if kind0 == kind1 => {}
            (Stream(t0), Stream(t1)) => self.unify(*t0, *t1),
            (Array(t0, sh0), Array(t1, sh1)) if sh0.dims.len() == sh1.dims.len() => {
                self.unify(*t0, *t1);
            }
            (Fun(ts0, t0), Fun(ts1, t1)) if ts0.len() == ts1.len() => {
                for (arg1, arg2) in ts0.iter().zip(ts1.iter()) {
                    self.unify(*arg1, *arg2);
                }
                self.unify(*t0, *t1);
            }
            (Tuple(ts0), Tuple(ts1)) if ts0.len() == ts1.len() => {
                for (t0, t1) in ts0.iter().zip(ts1.iter()) {
                    self.unify(*t0, *t1);
                }
            }
            (Struct(fs0), Struct(fs1)) => {
                for (f0, t0) in fs0 {
                    if let Some(t1) = fs1.get(f0) {
                        self.unify(*t0, *t1);
                    }
                }
            }
            _ => {
                let loc = self.loc;
                if matches!(loc, Loc::Fake) {
                    tracing::trace!(
                        "[Type mismatch in generated code]: {} != {}",
                        self.hir.pretty(&t0, self.info),
                        self.hir.pretty(&t1, self.info),
                    );
                }
                self.diags.intern(Error::TypeMismatch {
                    lhs: t0,
                    rhs: t1,
                    loc,
                })
            }
        }
        tracing::trace!(
            "unify(AFTER)({}, {})",
            self.hir.pretty(&t0, self.info),
            self.hir.pretty(&t1, self.info),
        );
    }
}
