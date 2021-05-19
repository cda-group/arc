use proc_macro as pm;
use quote::quote;

pub(crate) fn rewrite(_: syn::AttributeArgs, item: syn::ItemStruct) -> pm::TokenStream {
    {
        cfg_if::cfg_if! {
            if #[cfg(feature = "backend_arcon")] {
                let mut has_key = false;
                let mut item = item;
                item.fields.iter_mut().for_each(|field| {
                    let attr = crate::prost::ty_to_prost_attr(&field.ty, None);
                    field.attrs.push(attr);
                    let ident = field
                        .ident
                        .clone()
                        .expect("#[arcorn::rewrite] expects structs to have named fields");
                    if ident == "key" {
                        has_key = true;
                    }
                });
                let attr = if has_key {
                    quote!(#[arcon(reliable_ser_id = 13, version = 1, keys = "key")])
                } else {
                    quote!(#[arcon(reliable_ser_id = 13, version = 1)])
                };
                quote!(
                    #[derive(
                        Clone,
                        arc_script::arcorn::prost::Message,
                        arc_script::arcorn::arcon::prelude::Arcon,
                        arc_script::arcorn::derive_more::From,
                        arc_script::arcorn::derive_more::Constructor
                    )]
                    #attr
                    #item
                )
            } else {
                quote! {
                    #[derive(
                        Clone,
                        arc_script::arcorn::derive_more::From,
                        arc_script::arcorn::derive_more::Constructor
                    )]
                    #item
                }
            }
        }
    }
    .into()
}
