use super::Context;
use crate::compiler::ast;
use crate::compiler::hir;
use crate::compiler::hir::from::lower::resolve;
use crate::compiler::info::diags::Error;
use crate::compiler::info::files::Loc;

/// A bare-path (which is not the operand of a call expression) can resolve into different things.
pub(super) fn lower(path: &ast::Path, loc: Option<Loc>, ctx: &mut Context) -> hir::ExprKind {
    match ctx.res.resolve(path, ctx.info).unwrap() {
        resolve::DeclKind::Item(x, kind) => match kind {
            resolve::ItemDeclKind::Alias | resolve::ItemDeclKind::Enum => {
                ctx.info
                    .diags
                    .intern(Error::TypeInValuePosition { loc });
                hir::ExprKind::Err
            }
            resolve::ItemDeclKind::Fun => hir::ExprKind::Item(x),
            resolve::ItemDeclKind::Task => hir::ExprKind::Item(x),
            resolve::ItemDeclKind::Extern => hir::ExprKind::Item(x),
        },
        resolve::DeclKind::Var(x) => hir::ExprKind::Var(x),
    }
}
