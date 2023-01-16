use proc_macro as pm;
use quote::quote;

#[allow(unused)]
pub(crate) fn rewrite(attr: syn::AttributeArgs, mut item: syn::ItemEnum) -> pm::TokenStream {
    item.vis = syn::parse_quote!(pub);
    item.generics.params.iter_mut().for_each(|g| {
        if let syn::GenericParam::Type(t) = g {
            t.bounds.push(syn::parse_quote!(Data))
        }
    });

    let (impl_generics, type_generics, where_clause) = item.generics.split_for_impl();

    let variant_id = item.variants.iter().map(|v| &v.ident).collect::<Vec<_>>();

    quote!(
        use dataflow::prelude::*;

        #[derive(Clone, Debug, Unpin, Serialize, Deserialize)]
        #[serde(bound = "")]
        #item
    )
    .into()
}
