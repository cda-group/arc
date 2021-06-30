use proc_macro as pm;
use quote::quote;

pub(crate) fn rewrite(_: syn::AttributeArgs, item: syn::ItemStruct) -> pm::TokenStream {
    quote!(
        #[derive(
            Clone,
            Debug,
            arc_script::arcorn::derive_more::From,
            arc_script::arcorn::derive_more::Constructor
        )]
        #item
    )
    .into()
}
