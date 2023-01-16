use proc_macro as pm;

pub(crate) fn rewrite(_attr: syn::AttributeArgs, mut item: syn::ItemFn) -> pm::TokenStream {
    item.sig.generics.params.iter_mut().for_each(|p| {
        if let syn::GenericParam::Type(p) = p {
            p.bounds.push(syn::parse_quote!(Data));
        }
    });
    quote::quote!(#item).into()
}
