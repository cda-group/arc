use crate::compiler::hir::{
    self, BinOpKind, Dim, DimKind, Expr, ExprKind, Fun, LitKind, Param, ScalarKind, Shape, Type,
    TypeKind, UnOpKind, HIR,
};
use crate::compiler::info::diags::{Diagnostic, Error, Warning};
use crate::compiler::info::types::union::Union;
use crate::compiler::info::types::TypeId;

use super::Context;

pub(crate) trait Unify<A, B> {
    /// Unifies A with B
    fn unify(&mut self, a: A, b: B);
}

/// Unifies a type variable `a` with a type `b`.
impl<T: Into<Type>> Unify<TypeId, T> for Context<'_> {
    fn unify(&mut self, tv0: TypeId, ty1: T) {
        let tv1 = self.info.types.intern(ty1.into());
        self.unify(tv0, tv1);
    }
}

impl Unify<TypeId, TypeId> for Context<'_> {
    /// Unifies two polytypes, i.e., types which may contain type variables
    fn unify(&mut self, tv0: TypeId, tv1: TypeId) {
        use TypeKind::*;
        let ty0 = self.info.types.resolve(tv0);
        let ty1 = self.info.types.resolve(tv1);
        match (&ty0.kind, &ty1.kind) {
            // Unify Unknown types
            (Unknown, Unknown) => self.info.types.union(tv0, tv1),
            (Unknown, _) => self.info.types.union(tv0, ty1),
            (_, Unknown) => self.info.types.union(tv1, ty0),
            // Zip other types and unify their inner Unknown types
            (Nominal(x0), Nominal(x1)) if x0 == x1 => {}
            (Scalar(kind0), Scalar(kind1)) if kind0 == kind1 => {}
            (Optional(tv0), Optional(tv1)) => self.unify(*tv0, *tv1),
            (Boxed(tv0), Boxed(tv1)) => self.unify(*tv0, *tv1),
            (Stream(tv0), Stream(tv1)) => self.unify(*tv0, *tv1),
            (Set(tv0), Set(tv1)) => self.unify(*tv0, *tv1),
            (Vector(tv0), Vector(tv1)) => self.unify(*tv0, *tv1),
            (Array(tv0, sh0), Array(tv1, sh1)) if sh0.dims.len() == sh1.dims.len() => {
                self.unify(*tv0, *tv1);
            }
            (Fun(tvs0, tv0), Fun(tvs1, tv1)) if tvs0.len() == tvs1.len() => {
                for (arg1, arg2) in tvs0.iter().zip(tvs1.iter()) {
                    self.unify(*arg1, *arg2);
                }
                self.unify(*tv0, *tv1);
            }
            (Tuple(tvs0), Tuple(tvs1)) if tvs0.len() == tvs1.len() => {
                for (tv0, tv1) in tvs0.iter().zip(tvs1.iter()) {
                    self.unify(*tv0, *tv1);
                }
            }
            (Struct(fs0), Struct(fs1)) => {
                for (f0, tv0) in fs0 {
                    if let Some(tv1) = fs1.get(f0) {
                        self.unify(*tv0, *tv1);
                    }
                }
            }
            (Map(tv00, tv01), Map(tv10, tv11)) => {
                self.unify(*tv00, *tv10);
                self.unify(*tv01, *tv11);
            }
            _ => self.info.diags.intern(Error::TypeMismatch {
                lhs: tv0,
                rhs: tv1,
                loc: self.loc,
            }),
        }
    }
}
