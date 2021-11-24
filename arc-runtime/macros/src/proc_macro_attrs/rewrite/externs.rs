use crate::get_attr_val;
use proc_macro as pm;

#[cfg(feature = "legacy")]
pub(crate) fn rewrite(_attr: syn::AttributeArgs, item: syn::ItemFn) -> pm::TokenStream {
    quote::quote!(#item).into()
}

#[cfg(not(feature = "legacy"))]
pub(crate) fn rewrite(attr: syn::AttributeArgs, mut item: syn::ItemFn) -> pm::TokenStream {
    let unmangled = get_attr_val("unmangled", &attr);
    let params = item.sig.inputs.iter().map(|arg| {
        if let syn::FnArg::Typed(p) = arg {
            &p.pat
        } else {
            unreachable!()
        }
    });
    item.sig.abi = None;
    item.block = syn::parse_quote!({ #unmangled(#(#params),*) });
    quote::quote!(#item).into()
}
