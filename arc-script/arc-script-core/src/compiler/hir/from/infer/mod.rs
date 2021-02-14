use crate::compiler::hir::{
    self, BinOpKind, Dim, DimKind, Expr, ExprKind, Fun, Item, LitKind, Param, ScalarKind, Shape,
    Type, TypeKind, UnOpKind, HIR,
};
use crate::compiler::hir::{Name, Path};
use crate::compiler::info::files::Loc;
use crate::compiler::info::Info;
use arc_script_core_shared::OrdMap;
use arc_script_core_shared::Map;
use arc_script_core_shared::New;

pub(crate) mod constrain;
pub(crate) mod equality;
pub(crate) mod subtype;
pub(crate) mod unify;

use constrain::Constrain;

/// Context used during type inference.
#[derive(New)]
pub(crate) struct Context<'i> {
    defs: &'i OrdMap<Path, Item>,
    info: &'i mut Info,
    env: &'i mut Map<Name, Param>,
    loc: Option<Loc>,
}

impl HIR {
    /// Infers the types of all type variables in the HIR.
    pub(crate) fn infer(&self, info: &mut Info) {
        self.items.iter().for_each(|item| {
            let item = self.defs.get(item).unwrap();
            let env = &mut Map::default();
            let ctx = &mut Context::new(&self.defs, info, env, item.loc);
            item.constrain(ctx)
        });
    }
}
