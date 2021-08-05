use crate::hir;
use crate::hir::infer::unify::Unify;
use crate::hir::utils::SortFields;
use crate::info::diags::{Diagnostic, Error, Warning};
use crate::info::types::Type;

use arc_script_compiler_shared::itertools::Itertools;
use arc_script_compiler_shared::map;
use arc_script_compiler_shared::VecMap;

use super::Context;

use arc_script_compiler_shared::get;

pub(crate) trait Generalise {
    fn generalise(&self, ctx: &mut Context<'_>) -> hir::Type;
}
// /// Replace all free-type variables with quantifiers
// fn generalise(&self, ctx: &mut Context<'_>) {
//     // Assume all top-level functions are already generalised
// }

impl Generalise for hir::Item {
    /// Replace all quantifiers with free-type variables
    #[rustfmt::skip]
    fn generalise(&self, ctx: &mut Context<'_>) -> hir::Type {
        match &self.kind {
            hir::ItemKind::Alias(_)   => todo!(),
            hir::ItemKind::Enum(_)    => todo!(),
            hir::ItemKind::Fun(item)  => todo!(),
            hir::ItemKind::Task(_)    => todo!(),
            hir::ItemKind::Extern(_)  => todo!(),
            hir::ItemKind::Variant(_) => todo!(),
        }
    }
}

impl Generalise for hir::Fun {
    /// Replace all quantifiers with free-type variables
    fn generalise(&self, ctx: &mut Context<'_>) -> hir::Type {
        let bt = self.body.generalise(ctx);
        ctx.types.intern(hir::TypeKind::Fun(its, ot)).into()
    }
}

impl Generalise for hir::Block {
    fn generalise(&self, ctx: &mut Context<'_>) -> Type {
        for stmt in &self.stmts {
            stmt.generalise(ctx);
        }
        self.generalise(ctx)
    }
}

impl Generalise for hir::Stmt {
    fn generalise(&self, ctx: &mut Context<'_>) -> Type {
        todo!()
    }
}

impl Generalise for hir::Type {
    fn generalise(&self, ctx: &mut Context<'_>) -> hir::Type {}
}
