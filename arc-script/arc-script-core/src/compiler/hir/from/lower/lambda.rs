use super::Context;
use crate::compiler::ast;
use crate::compiler::hir;
use crate::compiler::hir::{Expr, ExprKind};
use crate::compiler::info::files::Loc;
use crate::compiler::info::paths::PathId;
use crate::compiler::shared::Lower;

// Lambdas must for now be pure (cannot capture anything)
pub(super) fn lower(
    params: &[ast::Param],
    body: &ast::Expr,
    loc: Option<Loc>,
    ctx: &mut Context<'_>,
) -> hir::ExprKind {
    ctx.res.stack.push_frame();

    let (name, path) = ctx.info.fresh_name_path();
    let (ps, cases) = super::pattern::lower_params(params, ctx);
    let body: Expr = body.lower(ctx);

    let body = super::pattern::fold_cases(body, None, cases);

    let tv = ctx.info.types.fresh();
    let rtv = ctx.info.types.fresh();

    let item = hir::Item::new(
        hir::ItemKind::Fun(hir::Fun::new(name, ps, body, tv, rtv)),
        loc,
    );

    ctx.hir.defs.insert(path, item);
    ctx.hir.items.push(path);

    ctx.res.stack.pop_frame();

    hir::ExprKind::Item(path)
}
