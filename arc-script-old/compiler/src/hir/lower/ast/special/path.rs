use super::Context;

use crate::ast;
use crate::hir;
use crate::hir::lower::ast::resolve;
use crate::info::diags::Error;
use crate::info::files::Loc;

use arc_script_compiler_shared::Lower;

/// A bare-path (which is not the operand of a call expression) can resolve into different things.
pub(crate) fn lower_expr_path(
    x: &ast::Path,
    ts: &Option<Vec<ast::Type>>,
    loc: Loc,
    ctx: &mut Context<'_>,
) -> hir::Var {
    if let Some(kind) = ctx.res.resolve(x, ctx.info) {
        match kind {
            resolve::DeclKind::Item(x, kind) => match kind {
                resolve::ItemDeclKind::Variant
                | resolve::ItemDeclKind::Method
                | resolve::ItemDeclKind::Alias
                | resolve::ItemDeclKind::Enum => {
                    ctx.diags.intern(Error::TypeInValuePosition { loc });
                    ctx.new_var_err(loc)
                }
                resolve::ItemDeclKind::Fun
                | resolve::ItemDeclKind::Fun
                | resolve::ItemDeclKind::Task
                | resolve::ItemDeclKind::ExternFun
                | resolve::ItemDeclKind::ExternType
                | resolve::ItemDeclKind::Global => ctx
                    .new_expr_with_loc(hir::ExprKind::Item(x), loc)
                    .into_ssa(ctx),
            },
            resolve::DeclKind::Var(x, kind) => {
                //                 if let Some((x, kind)) = ctx.res.stack.resolve(x) {
                ctx.new_var(x, kind)
                //                 } else {
                //                     ctx.create_var_err(loc)
                //                 }
            }
        }
    } else {
        ctx.new_var_err(loc)
    }
}

pub(crate) fn lower_type_path(
    x: &ast::Path,
    ts: &Option<Vec<ast::Type>>,
    loc: Loc,
    ctx: &mut Context<'_>,
) -> hir::TypeKind {
    if let Some(kind) = ctx.res.resolve(x, ctx.info) {
        match kind {
            resolve::DeclKind::Item(x, kind) => match kind {
                resolve::ItemDeclKind::Enum
                | resolve::ItemDeclKind::Alias
                | resolve::ItemDeclKind::ExternType => hir::TypeKind::Nominal(x),
                resolve::ItemDeclKind::Fun
                | resolve::ItemDeclKind::Task
                | resolve::ItemDeclKind::ExternFun
                | resolve::ItemDeclKind::Variant
                | resolve::ItemDeclKind::Method
                | resolve::ItemDeclKind::Global => {
                    ctx.diags.intern(Error::ValueInTypePosition { loc });
                    hir::TypeKind::Err
                }
            },
            resolve::DeclKind::Var(_, _) => {
                ctx.diags.intern(Error::ValueInTypePosition { loc });
                hir::TypeKind::Err
            }
        }
    } else {
        hir::TypeKind::Err
    }
}

pub(crate) fn lower_is_path(x: &ast::Path, e: &ast::Expr, ctx: &mut Context<'_>) -> hir::ExprKind {
    lower_variant_path(x, ctx)
        .map(|x| hir::ExprKind::Is(x, e.lower(ctx)))
        .unwrap_or(hir::ExprKind::Err)
}

pub(crate) fn lower_unwrap_path(
    x: &ast::Path,
    e: &ast::Expr,
    ctx: &mut Context<'_>,
) -> hir::ExprKind {
    lower_variant_path(x, ctx)
        .map(|x| hir::ExprKind::Unwrap(x, e.lower(ctx)))
        .unwrap_or(hir::ExprKind::Err)
}

pub(crate) fn lower_enwrap_path(
    x: &ast::Path,
    e: &ast::Expr,
    ctx: &mut Context<'_>,
) -> hir::ExprKind {
    lower_variant_path(x, ctx)
        .map(|x| hir::ExprKind::Enwrap(x, e.lower(ctx)))
        .unwrap_or(hir::ExprKind::Err)
}

pub(crate) fn lower_variant_path(x: &ast::Path, ctx: &mut Context<'_>) -> Option<hir::Path> {
    if let Some(kind) = ctx.res.resolve(x, ctx.info) {
        if let resolve::DeclKind::Item(x, kind) = kind {
            if let resolve::ItemDeclKind::Variant = kind {
                return Some(x);
            }
        }
    }
    ctx.info
        .diags
        .intern(Error::PathIsNotVariant { loc: x.loc });
    None
}

/// Lowers a `emit e` into a `emit crate::path::to::Task::OInterface::__(e)` in the context of a
/// task with an untagged port.
pub(crate) fn lower_emit(e0: &ast::Expr, ctx: &mut Context<'_>) -> hir::ExprKind {
    let v0 = e0.lower(ctx);
    if let Some(enum_path) = ctx.generated_ointerface_interior {
        let dummy_x = ctx.names.common.dummy;
        let variant_path = ctx.paths.intern_child(enum_path, dummy_x);
        let v1 = ctx.new_expr_enwrap(variant_path.into(), v0).into_ssa(ctx);
        hir::ExprKind::Emit(v1)
    } else {
        hir::ExprKind::Emit(v0)
    }
}
