use super::Context;
use crate::ast;
use crate::hir;
use crate::hir::Expr;
use crate::hir::FunKind::Free;
use crate::hir::ScopeKind;
use crate::info::files::Loc;
use arc_script_compiler_shared::Lower;

// Lambdas must for now be pure (cannot capture anything)
pub(crate) fn lower(
    params: &[ast::Param],
    e: &ast::Expr,
    loc: Loc,
    ctx: &mut Context<'_>,
) -> hir::ExprKind {
    ctx.res.stack.push_frame();

    let path = ctx.fresh_path();
    let (params, cases) = super::pattern::lower_params(params, ScopeKind::Local, ctx);

    let mut body = if let ast::ExprKind::Block(body) = ctx.ast.exprs.resolve(e.id) {
        body.lower(ctx)
    } else {
        ctx.new_implicit_block(e)
    };

    body.prepend_stmts(super::pattern::cases_to_stmts(cases));

    let t = ctx.types.fresh();
    let rt = ctx.types.fresh();

    let item = hir::Item::new(
        hir::ItemKind::Fun(hir::Fun::new(path, Free, params, body, t, rt)),
        loc,
    );

    ctx.hir.intern(path, item);
    ctx.hir.namespace.push(path.id);

    ctx.res.stack.pop_frame();

    hir::ExprKind::Item(path)
}
