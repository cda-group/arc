use super::Context;
use crate::compiler::ast;
use crate::compiler::hir;
use crate::compiler::hir::from::lower::resolve::DeclKind::Item;
use crate::compiler::hir::from::lower::resolve::ItemDeclKind::*;
use crate::compiler::info::diags::Error;

use arc_script_core_shared::Lower;

/// Call-expression are ambiguous and need to be analyzed in order to determine
/// their meaning. In particular, the expression `foo(1)` could either be:
/// * An enum-variant constructor, if `foo` resolves to an enum declaration.
/// * A function call, if `foo` resolves to a variable or function declaration.
/// Since functions are values, we only need to consider the enum-variant case.
pub(crate) fn lower(expr: &ast::Expr, args: &[ast::Expr], ctx: &mut Context<'_>) -> hir::ExprKind {
    if let ast::ExprKind::Path(path) = ctx.ast.exprs.resolve(expr.id) {
        if let Some(decl) = ctx.res.resolve(path, ctx.info) {
            if let Item(path, Variant) = decl {
                if let [arg] = args {
                    return hir::ExprKind::Enwrap(path, arg.lower(ctx).into());
                } else {
                    ctx.info.diags.intern(Error::VariantWrongArity { path });
                    return hir::ExprKind::Err;
                }
            }
        } else {
            return hir::ExprKind::Err;
        }
    };
    // If the path does not point to a Variant, then it might be a function call.
    // Since functions are values, they process it separately.
    hir::ExprKind::Call(expr.lower(ctx).into(), args.lower(ctx))
}
