use crate::compiler::arcorn::lower;
use crate::compiler::arcorn::Arcorn;
use crate::compiler::hir::HIR;
use crate::compiler::info::Info;
use arc_script_core_shared::Lower;
use arc_script_core_shared::Map;

use tracing::instrument;

impl Arcorn {
    #[instrument(name = "HIR => Arcorn", level = "debug", skip(hir, info))]
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
