use crate::compiler::ast;
use crate::compiler::hir;
use crate::compiler::hir::FunKind::Global;
use crate::compiler::hir::Name;
use crate::compiler::hir::Path;
use crate::compiler::info;
use crate::compiler::info::diags::Error;
use crate::compiler::info::files::Loc;
use crate::compiler::info::types::TypeId;

use arc_script_core_shared::itertools::Itertools;
use arc_script_core_shared::map;
use arc_script_core_shared::Lower;
use arc_script_core_shared::New;
use arc_script_core_shared::VecMap;

use super::Context;

pub(crate) fn lower_pipe(
    e0: &ast::Expr,
    e1: &ast::Expr,
    loc: Loc,
    ctx: &mut Context<'_>,
) -> hir::ExprKind {
    hir::ExprKind::Call(e1.lower(ctx).into(), vec![e0.lower(ctx)])
}

pub(crate) fn lower_not_in(
    e0: &ast::Expr,
    e1: &ast::Expr,
    loc: Loc,
    ctx: &mut Context<'_>,
) -> hir::ExprKind {
    hir::ExprKind::UnOp(
        hir::UnOp::syn(hir::UnOpKind::Not),
        hir::Expr::new(
            hir::ExprKind::BinOp(
                e0.lower(ctx).into(),
                hir::BinOp::syn(hir::BinOpKind::In),
                e1.lower(ctx).into(),
            ),
            ctx.info.types.fresh(),
            loc,
        )
        .into(),
    )
}

pub(crate) fn lower_del(e: &ast::Expr, loc: Loc, ctx: &mut Context<'_>) -> hir::ExprKind {
    if let ast::ExprKind::Select(e0, es) = ctx.ast.exprs.resolve(e.id) {
        if let [e1] = es.as_slice() {
            hir::ExprKind::Del(e0.lower(ctx).into(), e1.lower(ctx).into())
        } else {
            ctx.info.diags.intern(Error::MultipleSelectors { loc });
            hir::ExprKind::Err
        }
    } else {
        ctx.info.diags.intern(Error::ExpectedSelector { loc });
        hir::ExprKind::Err
    }
}

pub(crate) fn lower_add(e: &ast::Expr, loc: Loc, ctx: &mut Context<'_>) -> hir::ExprKind {
    if let ast::ExprKind::Select(e0, es) = ctx.ast.exprs.resolve(e.id) {
        if let [e1] = es.as_slice() {
            hir::ExprKind::Add(e0.lower(ctx).into(), e1.lower(ctx).into())
        } else {
            ctx.info.diags.intern(Error::MultipleSelectors { loc });
            hir::ExprKind::Err
        }
    } else {
        ctx.info.diags.intern(Error::ExpectedSelector { loc });
        hir::ExprKind::Err
    }
}

/// Lowers a `e0 after e1` into a `{val: e0, dur: e1}`
pub(crate) fn lower_after<'i, O, I: Lower<O, Context<'i>>>(
    val: &I,
    dur: &I,
    ctx: &mut Context<'i>,
) -> VecMap<Name, O> {
    lower_pair(val, dur, ctx.info.names.common.dur, ctx)
}

/// Lowers a `e0 by e1` into a `{val: e0, key: e1}`
pub(crate) fn lower_by<'i, O, I: Lower<O, Context<'i>>>(
    val: &I,
    key: &I,
    ctx: &mut Context<'i>,
) -> VecMap<Name, O> {
    lower_pair(val, key, ctx.info.names.common.key, ctx)
}

/// Lowers a `e0 <key> e1` into a `{val: e0, <key>: e1}`
pub(crate) fn lower_pair<'i, O, I: Lower<O, Context<'i>>>(
    val: &I,
    key: &I,
    key_name: impl Into<Name>,
    ctx: &mut Context<'i>,
) -> VecMap<Name, O> {
    let val = val.lower(ctx);
    let key = key.lower(ctx);
    let val_name = ctx.info.names.common.val.into();
    vec![(val_name, val), (key_name.into(), key)]
        .into_iter()
        .collect()
}
