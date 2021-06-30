//! Codegen for enums

use proc_macro as pm;
use proc_macro2 as pm2;
use quote::quote;

#[allow(unused)]
pub(crate) fn rewrite(_: syn::AttributeArgs, mut enum_item: syn::ItemEnum) -> pm::TokenStream {
    let enum_ident = syn::Ident::new(&format!("Enum{}", enum_item.ident), pm2::Span::call_site());
    let struct_ident = std::mem::replace(&mut enum_item.ident, enum_ident.clone());
    let appendix = quote! {
        use #enum_ident::*;
        impl #enum_ident {
            pub fn wrap(self) -> #struct_ident {
                #struct_ident { this: Some(self) }
            }
        }
    };

    quote!(
        #[derive(Clone, Debug)]
        pub struct #struct_ident {
            pub this: Option<#enum_ident>
        }
        #[derive(Clone, Debug)]
        #enum_item
        #appendix
    )
    .into()
}
