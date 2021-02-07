/// Module for doing additional checks on the typed HIR.
pub(crate) mod check;
/// Module for inferring types of the HIR.
pub(crate) mod infer;
/// Module for desugaring the AST into the HIR.
pub(crate) mod lower;

pub(crate) mod transform;

use crate::compiler::ast::AST;
use crate::compiler::hir;
use crate::compiler::hir::HIR;
use crate::compiler::info::Info;
use crate::compiler::shared::Lower;

/// Converts an AST into a HIR by doing the following:
/// * Resolve names and paths
/// * Desugar syntactic abstractions
/// * Infer types
impl HIR {
    pub(crate) fn from(ast: &AST, info: &mut Info) -> HIR {
        let mut hir = ast.lower(info);
        tracing::debug!("{}", hir::pretty(&hir, &hir, info));
        tracing::debug!("{}", hir.debug(info));
        hir.infer(info);
        hir.check(info);
        hir
    }
}
