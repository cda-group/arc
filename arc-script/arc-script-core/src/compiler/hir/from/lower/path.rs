use super::Context;
use crate::compiler::ast;
use crate::compiler::hir;
use crate::compiler::hir::from::lower::resolve;
use crate::compiler::info::diags::Error;
use crate::compiler::info::files::Loc;
use crate::compiler::shared::Lower;

use resolve::DeclKind::*;
use resolve::ItemDeclKind::*;

/// A bare-path (which is not the operand of a call expression) can resolve into different things.
pub(super) fn lower_path_expr(
    path: &ast::Path,
    loc: Option<Loc>,
    ctx: &mut Context<'_>,
) -> hir::ExprKind {
    match ctx.res.resolve(path, ctx.info).unwrap() {
        Item(x, kind) => match kind {
            Variant | Alias | Enum => {
                ctx.info.diags.intern(Error::TypeInValuePosition { loc });
                hir::ExprKind::Err
            }
            Fun | Task | Extern | State => hir::ExprKind::Item(x),
        },
        Var(x) => hir::ExprKind::Var(x),
    }
}

pub(super) fn lower_is(path: &ast::Path, expr: &ast::Expr, ctx: &mut Context<'_>) -> hir::ExprKind {
    lower_variant(path, ctx)
        .map(|path| hir::ExprKind::Is(path, expr.lower(ctx).into()))
        .unwrap_or(hir::ExprKind::Err)
}

pub(super) fn lower_unwrap(
    path: &ast::Path,
    expr: &ast::Expr,
    ctx: &mut Context<'_>,
) -> hir::ExprKind {
    lower_variant(path, ctx)
        .map(|path| hir::ExprKind::Unwrap(path, expr.lower(ctx).into()))
        .unwrap_or(hir::ExprKind::Err)
}

pub(super) fn lower_enwrap(
    path: &ast::Path,
    expr: &ast::Expr,
    ctx: &mut Context<'_>,
) -> hir::ExprKind {
    lower_variant(path, ctx)
        .map(|path| hir::ExprKind::Enwrap(path, expr.lower(ctx).into()))
        .unwrap_or(hir::ExprKind::Err)
}

pub(super) fn lower_variant(path: &ast::Path, ctx: &mut Context<'_>) -> Option<hir::Path> {
    if let Item(x, Variant) = ctx.res.resolve(path, ctx.info).unwrap() {
        Some(x)
    } else {
        ctx.info
            .diags
            .intern(Error::PathIsNotVariant { loc: path.loc });
        None
    }
}
