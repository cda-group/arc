mod lower_dfg;
mod lower_hir;
pub(crate) mod lowerings {
    pub(crate) mod structs;
}

use crate::compiler::hir;
use crate::compiler::hir::HIR;
use crate::compiler::info::Info;
use crate::compiler::rust::Rust;
use arc_script_core_shared::{Map, New};

#[derive(Debug, New)]
pub(crate) struct Context<'i> {
    pub(crate) info: &'i Info,
    pub(crate) hir: &'i HIR,
    /// Already mangled (root) type-variables.
    pub(crate) mangled_names: Map<hir::TypeId, String>,
    pub(crate) mangled_defs: Map<syn::Ident, proc_macro2::TokenStream>,
}
