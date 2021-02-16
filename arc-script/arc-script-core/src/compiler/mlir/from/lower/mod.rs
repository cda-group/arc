pub(super) mod lower_dfg;
pub(super) mod lower_hir;

/// Module for converting the [`crate::repr::hir::HIR`] into SSA form.
pub(crate) mod ssa;

use crate::compiler::hir::HIR;
use crate::compiler::info::Info;

use arc_script_core_shared::New;

#[derive(New)]
pub(crate) struct Context<'i> {
    hir: &'i HIR,
    info: &'i mut Info,
}
