use crate::compiler::ast;
use crate::compiler::hir;
use crate::compiler::hir::FunKind::Global;
use crate::compiler::hir::Name;
use crate::compiler::hir::Path;
use crate::compiler::info;
use crate::compiler::info::diags::Error;
use crate::compiler::info::files::Loc;
use crate::compiler::info::types::TypeId;

use arc_script_core_shared::map;
use arc_script_core_shared::Lower;
use arc_script_core_shared::New;
use arc_script_core_shared::VecMap;

use super::Context;

pub(crate) fn lower_pipe(
    e0: &ast::Expr,
    e1: &ast::Expr,
    loc: Option<Loc>,
    ctx: &mut Context<'_>,
) -> hir::ExprKind {
    hir::ExprKind::Call(e1.lower(ctx).into(), vec![e0.lower(ctx)])
}

pub(crate) fn lower_not_in(
    e0: &ast::Expr,
    e1: &ast::Expr,
    loc: Option<Loc>,
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

pub(crate) fn lower_del(e: &ast::Expr, loc: Option<Loc>, ctx: &mut Context<'_>) -> hir::ExprKind {
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

pub(crate) fn lower_add(e: &ast::Expr, loc: Option<Loc>, ctx: &mut Context<'_>) -> hir::ExprKind {
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
