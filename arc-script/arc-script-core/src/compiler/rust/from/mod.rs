mod lower;

use crate::compiler::hir::HIR;
use crate::compiler::info::Info;
use crate::compiler::rust::Rust;
use arc_script_core_shared::Lower;
use arc_script_core_shared::Map;

use tracing::instrument;

impl Rust {
    #[instrument(name = "HIR & Info => Rust", level = "debug", skip(hir, info))]
    pub(crate) fn from(hir: &HIR, info: &Info) -> Self {
        let buf = String::default();
        let mangled = Map::default();
        let ctx = &mut lower::Context::new(info, hir, buf, mangled);
        let items: proc_macro2::TokenStream = hir.lower(ctx);
        let file: syn::File = syn::parse_quote!(#items);
        let rust = Rust::new(file);
        rust
    }
}
