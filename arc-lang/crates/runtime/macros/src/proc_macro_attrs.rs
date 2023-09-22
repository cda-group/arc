use proc_macro as pm;

#[allow(unused)]
pub(crate) fn data(mut item: syn::DeriveInput) -> pm::TokenStream {
    quote::quote! {
        #[derive(Debug, Clone, Send, DeepClone, serde::Serialize, serde::Deserialize)]
        #[serde(crate = "runtime::prelude::serde")]
        #[serde(bound(deserialize = ""))]
        #item
    }
    .into()
}
