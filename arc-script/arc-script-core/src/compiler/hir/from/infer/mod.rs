use crate::compiler::hir::{
    self, BinOpKind, Dim, DimKind, Expr, ExprKind, Fun, Item, LitKind, Param, ScalarKind, Shape,
    Type, TypeKind, UnOpKind, HIR,
};
use crate::compiler::hir::{Name, Path};
use crate::compiler::info::diags::{Diagnostic, Error, Warning};
use crate::compiler::info::files::Loc;
use crate::compiler::info::types::union::Union;
use crate::compiler::info::types::{TypeId, TypeInterner};
use crate::compiler::info::Info;
use crate::compiler::shared::{Map, New, VecMap};

use ena::unify::{InPlace, NoError, UnifyKey, UnifyValue};

pub(crate) mod constrain;
pub(crate) mod equality;
pub(crate) mod subtype;
pub(crate) mod unify;

use constrain::Constrain;

/// Context used during type inference.
#[derive(New)]
pub(crate) struct Context<'i> {
    defs: &'i Map<Path, Item>,
    info: &'i mut Info,
    env: &'i mut Map<Name, Param>,
    loc: Option<Loc>,
}

impl HIR {
    /// Infers the types of all type variables in the HIR.
    pub fn infer(&self, info: &mut Info) {
        self.items.iter().for_each(|item| {
            let item = self.defs.get(item).unwrap();
            let env = &mut Map::new();
            let ctx = &mut Context::new(&self.defs, info, env, item.loc);
            item.constrain(ctx)
        });
    }
}