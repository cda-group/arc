use super::Context;
use crate::ast;
use crate::hir;
use arc_script_compiler_shared::Lower;

pub(crate) fn lower_if(
    e: &ast::Expr,
    b0: &ast::Block,
    b1: &Option<ast::Block>,
    ctx: &mut Context<'_>,
) -> hir::ExprKind {
    let e = e.lower(ctx);
    let b0 = b0.lower(ctx);
    let b1 = ctx.new_block_unit_if_none(b1);
    hir::ExprKind::If(e, b0, b1)
}

pub(crate) fn lower_if_assign(
    l: &ast::Assign,
    b0: &ast::Block,
    b1: &Option<ast::Block>,
    ctx: &mut Context<'_>,
) -> hir::Var {
    let cases = super::pattern::lower_assign(l, true, hir::ScopeKind::Local, ctx);
    let b0 = b0.lower(ctx);
    let b1 = ctx.new_block_unit_if_none(b1);
    let b1 = super::lift::lift(b1, ctx);
    let mut b = ctx.fold_cases(b0, b1, cases);
    // Flatten the block
    ctx.get_stmts().append(&mut b.stmts);
    b.var
}
