use crate::rust::lower;
use crate::rust::Rust;
use crate::hir::HIR;
use crate::info::Info;
use arc_script_compiler_shared::Lower;
use arc_script_compiler_shared::Map;

use tracing::instrument;

impl Rust {
    #[instrument(name = "HIR => Rust", level = "debug", skip(hir, info))]
    pub(crate) fn from(hir: &HIR, info: &mut Info) -> Self {
        let mangled_idents = Map::default();
        let mangled_defs = Map::default();
        let ctx = &mut lower::hir::Context::new(info, hir, mangled_idents, mangled_defs);
        let items: proc_macro2::TokenStream = hir.lower(ctx);
        tracing::debug!("\n{}", items);
        let file: syn::File = syn::parse_quote!(#items);
        Self::new(file)
    }
}
