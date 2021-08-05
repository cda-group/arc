use super::Context;
use crate::ast;
use crate::hir;
use crate::hir::lower::ast::resolve::DeclKind::Item;
use crate::hir::lower::ast::resolve::ItemDeclKind::*;
use crate::info::diags::Error;

use arc_script_compiler_shared::Lower;

/// Call-expression are ambiguous and need to be analyzed in order to determine
/// their meaning. In particular, the expression `foo(1)` could either be:
/// * An enum-variant constructor, if `foo` resolves to an enum declaration.
/// * A function call, if `foo` resolves to a variable or function declaration.
/// Since functions are values, we only need to consider the enum-variant case.
pub(crate) fn lower(expr: &ast::Expr, args: &[ast::Expr], ctx: &mut Context<'_>) -> hir::ExprKind {
    match ctx.ast.exprs.resolve(expr.id) {
        ast::ExprKind::Path(path, _) => match ctx.res.resolve(path, ctx.info) {
            Some(Item(path, Method)) => hir::ExprKind::SelfCall(path, args.lower(ctx)),
            Some(Item(path, Variant)) => {
                if let [arg] = args {
                    hir::ExprKind::Enwrap(path, arg.lower(ctx))
                } else {
                    ctx.diags.intern(Error::VariantWrongArity { path });
                    hir::ExprKind::Err
                }
            }
            None => hir::ExprKind::Err,
            _ => hir::ExprKind::Call(expr.lower(ctx), args.lower(ctx)),
        },
        _ => hir::ExprKind::Call(expr.lower(ctx), args.lower(ctx)),
    }
}
