use crate::compiler::hir::HIR;
use crate::compiler::info::Info;
use crate::compiler::mlir::lower;
use crate::compiler::mlir::MLIR;

use arc_script_core_shared::Lower;
use arc_script_core_shared::OrdMap;

use lower::hir::Context;

use tracing::instrument;

impl MLIR {
    #[instrument(name = "HIR => MLIR", level = "debug", skip(hir, info))]
    pub(crate) fn from(hir: &HIR, info: &mut Info) -> Self {
        let ops = Vec::new();
        let mut mlir = MLIR::default();
        let ctx = &mut Context::new(hir, &mut mlir, info, ops);
        hir.lower(ctx);
        mlir
    }
}
