use super::Context;
use crate::compiler::ast;
use crate::compiler::hir;
use crate::compiler::hir::lower::ast::resolve;

pub(crate) fn desugar(path: &ast::Path, ctx: &mut Context<'_>) -> hir::TypeKind {
    let decl = ctx.res.resolve(path, ctx.info).unwrap();
    if let resolve::DeclKind::Item(path, kind) = decl {
        match kind {
            resolve::ItemDeclKind::Enum => return hir::TypeKind::Nominal(path),
            resolve::ItemDeclKind::Alias => return hir::TypeKind::Nominal(path),
            _ => {}
        }
    }
    hir::TypeKind::Err
}
