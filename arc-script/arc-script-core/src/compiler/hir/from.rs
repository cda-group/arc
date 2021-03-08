use crate::compiler::ast::AST;
use crate::compiler::hir;
use crate::compiler::hir::HIR;
use crate::compiler::info::Info;
use arc_script_core_shared::Lower;

use tracing::instrument;

/// Converts an AST into a HIR by doing the following:
/// * Resolve names and paths
/// * Desugar syntactic abstractions
/// * Infer types
impl HIR {
    #[instrument(name = "AST & Info => HIR", level = "debug", skip(ast, info))]
    pub(crate) fn from(ast: &AST, info: &mut Info) -> Self {
        tracing::debug!("{}", info);
        let hir = ast.lower(info);
        tracing::debug!("{}", hir::pretty(&hir, &hir, info));
        tracing::debug!("Lowered: {}", hir.debug(info));
        hir.infer(info);
        tracing::debug!("Inferred: {}", hir.debug(info));
        hir.check(info);
        tracing::trace!("Checked: {}", hir.debug(info));
        hir
    }
}
