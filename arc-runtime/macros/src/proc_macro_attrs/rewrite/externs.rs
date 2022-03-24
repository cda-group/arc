use crate::get_attr_val;
use proc_macro as pm;

#[cfg(feature = "legacy")]
pub(crate) fn rewrite(_attr: syn::AttributeArgs, item: syn::ItemFn) -> pm::TokenStream {
    quote::quote!(#item).into()
}

#[cfg(not(feature = "legacy"))]
pub(crate) fn rewrite(attr: syn::AttributeArgs, mut item: syn::ItemFn) -> pm::TokenStream {
    let unmangled = get_attr_val("unmangled", &attr);
    if unmangled != item.sig.ident {
        let tys = item.sig.inputs.iter().map(|arg| {
            if let syn::FnArg::Typed(p) = arg {
                &p.ty
            } else {
                unreachable!()
            }
        });
        item.sig.abi = None;
        item.sig.inputs = syn::parse_quote!(x: (#(#tys,)*), ctx: Context);
        item.block = syn::parse_quote!({ #unmangled(x, ctx) });
        quote::quote!(#item).into()
    } else {
        quote::quote!().into()
    }
}
