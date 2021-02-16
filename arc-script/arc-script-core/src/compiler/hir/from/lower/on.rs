use super::Context;
use crate::compiler::ast;
use crate::compiler::hir;

/// Lowers an on-expression `on x`
pub(super) fn lower(_cases: &[ast::Case], _ctx: &mut Context<'_>) -> hir::ExprKind {
    todo!()
}
