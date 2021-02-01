mod lower_hir;
mod lower_dfg;
pub(crate) mod mangle;

use crate::compiler::dfg::DFG;
use crate::compiler::hir::HIR;
use crate::compiler::hir;
use crate::compiler::info::Info;
use crate::compiler::rust::Rust;
use crate::compiler::shared::{Lower, Map, New};

#[derive(Debug, New)]
pub(crate) struct Context<'i> {
    pub(crate) info: &'i Info,
    pub(crate) rust: &'i mut Rust,
    /// Buffer for name mangling
    pub(crate) buf: String,
    /// Already mangled (root) type-variables.
    pub(crate) mangled: Map<hir::TypeId, syn::Ident>,
}
