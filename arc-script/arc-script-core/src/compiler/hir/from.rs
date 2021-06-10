use crate::compiler::ast::AST;
use crate::compiler::hir;
use crate::compiler::hir::HIR;
use crate::compiler::info::Info;
use crate::compiler::hir::lower::ast::resolve;

use arc_script_core_shared::Lower;

use tracing::instrument;

/// Converts an AST into a HIR by doing the following:
/// * Resolve names and paths
/// * Desugar syntactic abstractions
/// * Infer types
impl HIR {
    #[instrument(name = "AST => HIR", level = "debug", skip(ast, info))]
    pub(crate) fn from(ast: &AST, info: &mut Info) -> Self {
        tracing::debug!("\n{:?}", info);
        let res = &mut resolve::Resolver::from(ast, info);
        let hir = ast.lower(res, info);
        tracing::debug!("\n{}", hir.pretty(&hir, info));
        if !info.mode.no_infer {
            tracing::debug!("Lowered: \n{}", hir.debug(info));
            hir.infer(info);
        }
        tracing::debug!("Inferred: \n{}", hir.debug(info));
        //         hir.check(info);
//         tracing::trace!("Checked: \n{}", hir.debug(info));
        hir
    }
}
