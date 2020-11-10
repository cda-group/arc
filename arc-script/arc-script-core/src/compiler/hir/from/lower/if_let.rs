use super::Context;
use crate::compiler::ast;
use crate::compiler::hir::{Expr, ExprKind, Param, ParamKind};
use crate::compiler::info::diags::Error;
use crate::compiler::info::names::NameId;
use crate::compiler::info::types::TypeId;
use crate::compiler::shared::Lower;

pub(super) fn lower(
    p: &ast::Pat,
    e0: &ast::Expr,
    e1: &ast::Expr,
    e2: &ast::Expr,
    ctx: &mut Context,
) -> Expr {
    let e0 = e0.lower(ctx);
    let clauses = super::pattern::lower_branching_pat_expr(p, e0, ctx);
    let e1 = e1.lower(ctx);
    let e2 = e2.lower(ctx);
    let e2 = super::lift::lift(e2, ctx);
    super::pattern::fold_cases(e1, Some(e2), clauses)
}
