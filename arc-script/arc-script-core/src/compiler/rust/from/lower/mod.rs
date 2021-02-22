mod lower_dfg;
mod lower_hir;
pub(crate) mod mangle;

use crate::compiler::hir;
use crate::compiler::hir::HIR;
use crate::compiler::info::Info;
use crate::compiler::rust::Rust;
use arc_script_core_shared::{Map, New};

#[derive(Debug, New)]
pub(crate) struct Context<'i> {
    pub(crate) info: &'i Info,
    pub(crate) hir: &'i HIR,
    /// Buffer for name mangling
    pub(crate) buf: String,
    /// Already mangled (root) type-variables.
    pub(crate) mangled: Map<hir::TypeId, syn::Ident>,
}
