use proc_macro as pm;
use proc_macro2 as pm2;
use quote::quote;

#[allow(unused)]
pub(crate) fn rewrite(_: syn::AttributeArgs, mut enum_item: syn::ItemEnum) -> pm::TokenStream {
    let enum_ident = syn::Ident::new(&format!("Enum{}", enum_item.ident), pm2::Span::call_site());
    let struct_ident = std::mem::replace(&mut enum_item.ident, enum_ident.clone());
    enum_item
        .variants
        .iter_mut()
        .enumerate()
        .for_each(|(tag, variant)| {
            assert!(
                variant.fields.len() == 1,
                "#[codegen::rewrite] expects variant fields to take exactly one argument"
            );
            {
                cfg_if::cfg_if! {
                    if #[cfg(feature = "backend_arcon")] {
                        let field = variant.fields.iter_mut().next().unwrap();
                        let attr = crate::prost::ty_to_prost_attr(&field.ty, Some(tag));
                        variant.attrs.push(attr);
                    }
                }
            }
        });
    let tags = enum_item
        .variants
        .iter()
        .enumerate()
        .map(|(i, _)| format!("{}", i))
        .collect::<Vec<_>>()
        .join(",");
    let tags = syn::LitStr::new(&tags, pm2::Span::call_site());
    let enum_ident_str = syn::LitStr::new(&enum_ident.to_string(), enum_ident.span());
    let appendix = quote! {
        use #enum_ident::*;
        impl #enum_ident {
            pub fn wrap(self) -> #struct_ident {
                #struct_ident { this: Some(self) }
            }
        }
    };
    {
        // NOTE: prost automatically derives Debug
        cfg_if::cfg_if! {
            if #[cfg(feature = "backend_arcon")] {
                quote! {
                    #[derive(
                        arc_script::codegen::prost::Message,
                        arc_script::codegen::arcon::prelude::Arcon,
                        Clone
                    )]
                    #[arcon(reliable_ser_id = 13, version = 1)]
                    pub struct #struct_ident {
                        #[prost(oneof = #enum_ident_str, tags = #tags)]
                        pub this: Option<#enum_ident>
                    }
                    #[derive(arc_script::codegen::prost::Oneof, Clone)]
                    #enum_item
                    #appendix
                }
            } else {
                quote! {
                    #[derive(Clone, Debug)]
                    pub struct #struct_ident {
                        pub this: Option<#enum_ident>
                    }
                    #[derive(Clone, Debug)]
                    #enum_item
                    #appendix
                }
            }
        }
    }
    .into()
}
