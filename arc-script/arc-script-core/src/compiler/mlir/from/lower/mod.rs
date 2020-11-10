pub(super) mod lower_dfg;
pub(super) mod lower_hir;

/// Module for converting the [`crate::repr::hir::HIR`] into SSA form.
pub(crate) mod ssa;

use crate::compiler::dfg::DFG;
use crate::compiler::hir::HIR;
use crate::compiler::info::Info;
use crate::compiler::mlir;
use crate::compiler::mlir::MLIR;
use crate::compiler::shared::{Lower, Map, New};

#[derive(New)]
pub(crate) struct Context<'i> {
    hir: &'i HIR,
    info: &'i mut Info,
}
