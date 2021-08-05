use crate::ast;
use crate::hir;
use crate::hir::FunKind::Free;
use crate::hir::Name;
use crate::hir::Path;
use crate::info;
use crate::info::diags::Error;
use crate::info::files::Loc;
use crate::info::types::TypeId;

use arc_script_compiler_shared::itertools::Itertools;
use arc_script_compiler_shared::map;
use arc_script_compiler_shared::Lower;
use arc_script_compiler_shared::New;
use arc_script_compiler_shared::VecMap;

use super::Context;

pub(crate) fn lower_pipe(
    e0: &ast::Expr,
    e1: &ast::Expr,
    loc: Loc,
    ctx: &mut Context<'_>,
) -> hir::ExprKind {
    hir::ExprKind::Call(e1.lower(ctx), vec![e0.lower(ctx)])
}

pub(crate) fn lower_notin(
    e0: &ast::Expr,
    e1: &ast::Expr,
    loc: Loc,
    ctx: &mut Context<'_>,
) -> hir::ExprKind {
    let v0 = e0.lower(ctx);
    let v1 = e1.lower(ctx);
    let kind = hir::ExprKind::BinOp(v0, hir::BinOp::syn(hir::BinOpKind::In), v1);
    let v = ctx.new_expr_with_loc(kind, loc).into_ssa(ctx);
    hir::ExprKind::UnOp(hir::UnOp::syn(hir::UnOpKind::Not), v)
}

/// Lowers a `e0 by e1` into a `{val: e0, key: e1}`
pub(crate) fn lower_by<'i, O, I: Lower<O, Context<'i>>>(
    val: &I,
    key: &I,
    ctx: &mut Context<'i>,
) -> VecMap<Name, O> {
    lower_pair(val, key, ctx.names.common.key, ctx)
}

/// Lowers a `e0 <key> e1` into a `{val: e0, <key>: e1}`
pub(crate) fn lower_pair<'i, O, I: Lower<O, Context<'i>>>(
    val: &I,
    key: &I,
    key_name: Name,
    ctx: &mut Context<'i>,
) -> VecMap<Name, O> {
    let val = val.lower(ctx);
    let key = key.lower(ctx);
    let val_name = ctx.names.common.val;
    vec![(val_name, val), (key_name, key)].into_iter().collect()
}
