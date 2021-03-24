use super::Context;
use crate::compiler::ast;
use crate::compiler::hir;
use crate::compiler::hir::lower::ast::resolve;
use crate::compiler::info::diags::Error;
use crate::compiler::info::files::Loc;
use arc_script_core_shared::Lower;

use resolve::DeclKind::*;
use resolve::ItemDeclKind::*;

/// A bare-path (which is not the operand of a call expression) can resolve into different things.
pub(crate) fn lower_path_expr(path: &ast::Path, loc: Loc, ctx: &mut Context<'_>) -> hir::ExprKind {
    ctx.res
        .resolve(path, ctx.info)
        .map(|decl| match decl {
            Item(x, kind) => match kind {
                Variant | Alias | Enum => {
                    ctx.info.diags.intern(Error::TypeInValuePosition { loc });
                    hir::ExprKind::Err
                }
                Fun | Task | Extern | State => hir::ExprKind::Item(x),
            },
            Var(x, kind) => hir::ExprKind::Var(x, kind),
        })
        .unwrap_or(hir::ExprKind::Err)
}

pub(crate) fn lower_is(path: &ast::Path, expr: &ast::Expr, ctx: &mut Context<'_>) -> hir::ExprKind {
    lower_variant(path, ctx)
        .map(|path| hir::ExprKind::Is(path, expr.lower(ctx).into()))
        .unwrap_or(hir::ExprKind::Err)
}

pub(crate) fn lower_unwrap(
    path: &ast::Path,
    expr: &ast::Expr,
    ctx: &mut Context<'_>,
) -> hir::ExprKind {
    lower_variant(path, ctx)
        .map(|path| hir::ExprKind::Unwrap(path, expr.lower(ctx).into()))
        .unwrap_or(hir::ExprKind::Err)
}

pub(crate) fn lower_enwrap(
    path: &ast::Path,
    expr: &ast::Expr,
    ctx: &mut Context<'_>,
) -> hir::ExprKind {
    lower_variant(path, ctx)
        .map(|path| hir::ExprKind::Enwrap(path, expr.lower(ctx).into()))
        .unwrap_or(hir::ExprKind::Err)
}

pub(crate) fn lower_variant(path: &ast::Path, ctx: &mut Context<'_>) -> Option<hir::Path> {
    if let Item(x, Variant) = ctx.res.resolve(path, ctx.info).unwrap() {
        Some(x)
    } else {
        ctx.info
            .diags
            .intern(Error::PathIsNotVariant { loc: path.loc });
        None
    }
}
