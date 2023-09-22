use proc_macro as pm;

pub(crate) fn rewrite(attr: syn::AttributeArgs, mut item: syn::ItemFn) -> pm::TokenStream {
    let unmangled: syn::Path = crate::utils::get_attr_val("unmangled", &attr);
    let ids = item.sig.inputs.iter().map(|arg| {
        if let syn::FnArg::Typed(p) = arg {
            p.pat.as_ref()
        } else {
            unreachable!()
        }
    });
    item.sig.abi = None;
    if item.sig.asyncness.is_some() {
        item.block = syn::parse_quote!({ #unmangled(#(#ids),*).await });
    } else {
        item.block = syn::parse_quote!({ #unmangled(#(#ids),*) });
    }
    quote::quote!(#item).into()
}
