use super::Context;
use crate::compiler::ast;
use crate::compiler::hir;
use crate::compiler::shared::Lower;

/// Lowers an on-expression `on x`
pub(super) fn lower(cases: &Vec<ast::Case>, ctx: &mut Context) -> hir::ExprKind {
    todo!()
}
