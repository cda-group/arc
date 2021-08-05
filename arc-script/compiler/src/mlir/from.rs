use crate::hir::HIR;
use crate::info::Info;
use crate::mlir::lower;
use crate::mlir::MLIR;

use arc_script_compiler_shared::Lower;
use arc_script_compiler_shared::OrdMap;

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
