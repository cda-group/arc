use crate::compiler::hir::{
    self, BinOpKind, Dim, DimKind, Expr, ExprKind, Fun, LitKind, Name, Param, ScalarKind, Shape,
    Type, TypeKind, UnOpKind, HIR,
};
use crate::compiler::info::diags::{Diagnostic, Error, Warning};
use crate::compiler::info::files::Loc;
use crate::compiler::info::names::NameId;
use crate::compiler::info::paths::PathId;
use crate::compiler::info::types::union::Union;
use crate::compiler::info::types::{TypeId, TypeInterner};
use crate::compiler::info::Info;
use crate::compiler::shared::{Map, VecMap};

use derive_more::Constructor;
use ena::unify::{InPlace, NoError, UnifyKey, UnifyValue};

use super::equality::Equal;
use super::Context;

trait Subtype<A, B> {
    /// Checks if `a <: b`
    fn subtype(&mut self, a: A, b: B) -> bool;
}

impl Subtype<Vec<TypeId>, Vec<TypeId>> for Context<'_> {
    fn subtype(&mut self, ts0: Vec<TypeId>, ts1: Vec<TypeId>) -> bool {
        (ts0.len() == ts1.len())
            && ts0
                .into_iter()
                .zip(ts1.into_iter())
                .all(|(t0, t1)| self.subtype(t0, t1))
    }
}

impl Subtype<VecMap<Name, TypeId>, VecMap<Name, TypeId>> for Context<'_> {
    fn subtype(&mut self, fs0: VecMap<Name, TypeId>, fs1: VecMap<Name, TypeId>) -> bool {
        (fs0.len() >= fs1.len())
            && fs1
                .into_iter()
                .all(|(x1, t1)| matches!(fs0.get(&x1), Some(t0) if self.subtype(*t0, t1)))
    }
}

impl Subtype<TypeId, TypeId> for Context<'_> {
    /// Returns `true` if `A` is a subtype of `B`.
    #[rustfmt::skip]
    fn subtype(&mut self, t0: TypeId, t1: TypeId) -> bool {
        use ScalarKind::*;
        use TypeKind::*;
        let ty0 = self.info.types.resolve(t0);
        let ty1 = self.info.types.resolve(t1);
        match (ty0.kind, ty1.kind) {
            (Fun(ts0, t0), Fun(ts1, t1)) => self.subtype(ts1, ts0) && self.subtype(t0, t1),
            (Scalar(Bot), _)             => true,
            (_, Scalar(Bot))             => false,
            (Struct(fs0), Struct(fs1))   => self.subtype(fs0, fs1),
            _                            => self.eq(t0, t1),
        }
    }
}
