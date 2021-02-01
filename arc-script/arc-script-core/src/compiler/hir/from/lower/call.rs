use super::{Context, Lower};
use crate::compiler::ast;
use crate::compiler::hir;
use crate::compiler::hir::from::lower::resolve;

/// Call-expression are ambiguous and need to be analyzed in order to determine
/// their meaning. In particular, the expression `foo(1)` could either be:
/// * An enum-variant constructor, if `foo` resolves to an enum declaration.
/// * A function call, if `foo` resolves to a variable or function declaration.
/// Since functions are values, we only need to consider the enum-variant case.
pub(super) fn lower(expr: &ast::Expr, args: &Vec<ast::Expr>, ctx: &mut Context) -> hir::ExprKind {
    if let ast::ExprKind::Path(path) = ctx.ast.exprs.resolve(expr.id) {
        let decl = ctx.res.resolve(path, ctx.info).unwrap();
        if let resolve::DeclKind::Item(path, resolve::ItemDeclKind::Enum) = decl {
            if let [arg] = args.as_slice() {
                let mut names = ctx.info.paths.resolve(path.id).clone();
                let variant = names.pop().unwrap();
                let id = ctx.info.paths.intern(names);
                let path = hir::Path::new(id, path.loc);
                return hir::ExprKind::Enwrap(path, variant, arg.lower(ctx).into());
            } else {
                panic!("[FIXME] Expected exactly one argument to variant constructor")
            }
        }
    };
    // If the path does not point to a Variant, then it might be a function call.
    // Since functions are values, they process it separately.
    hir::ExprKind::Call(expr.lower(ctx).into(), args.lower(ctx))
}
