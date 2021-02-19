use proc_macro as pm;
use proc_macro2 as pm2;
use quote::quote;

pub(super) fn execute(item: pm::TokenStream) -> pm::TokenStream {
    let item: syn::Item = syn::parse_macro_input!(item as syn::Item);
    match item {
        syn::Item::Enum(item) => rewrite_enum(item),
        syn::Item::Struct(item) => rewrite_struct(item),
        _ => panic!("#[arcorn::rewrite] expects enum or struct as input"),
    }
}

fn rewrite_enum(mut item: syn::ItemEnum) -> pm::TokenStream {
    item.variants
        .iter_mut()
        .enumerate()
        .for_each(|(tag, variant)| {
            assert!(
                variant.fields.len() == 1,
                "#[arcorn::rewrite] expects variant fields to take exactly one argument"
            );
            let field = variant.fields.iter().next().unwrap();
            variant.attrs.push(ty_to_prost_attr(&field.ty, Some(tag)))
        });
    let tags = item
        .variants
        .iter()
        .enumerate()
        .map(|(i, _)| format!("{}", i))
        .collect::<Vec<_>>()
        .join(",");
    let tags = syn::LitStr::new(&tags, pm2::Span::call_site());
    let enum_ident = syn::Ident::new(&format!("Enum{}", item.ident), pm2::Span::call_site());
    let struct_ident = std::mem::replace(&mut item.ident, enum_ident.clone());
    let enum_ident_str = syn::LitStr::new(&enum_ident.to_string(), enum_ident.span());
    quote!(
        #[derive(prost::Message, arcon::prelude::Arcon, Clone, arc_script::arcorn::derive_more::From)]
        #[arcon(reliable_ser_id = 13, version = 1)]
        struct #struct_ident {
            #[prost(oneof = #enum_ident_str, tags = #tags)]
            pub this: Option<#enum_ident>
        }
        #[derive(prost::Oneof, Clone)]
        #item
        use #enum_ident::*;
        impl #enum_ident {
            fn wrap(self) -> #struct_ident {
                #struct_ident { this: Some(self) }
            }
        }

    )
    .into()
}

fn rewrite_struct(mut item: syn::ItemStruct) -> pm::TokenStream {
    item.fields.iter_mut().for_each(|field| {
        field.attrs.push(ty_to_prost_attr(&field.ty, None));
    });
    quote!(
        #[derive(prost::Message, arcon::prelude::Arcon, Clone, arc_script::arcorn::derive_more::From)]
        #[arcon(reliable_ser_id = 13, version = 1)]
        #item
    )
    .into()
}

fn ty_to_prost_attr(ty: &syn::Type, tag: Option<usize>) -> syn::Attribute {
    let ty = match &ty {
        syn::Type::Path(ty) => {
            let ident = ty
                .path
                .get_ident()
                .expect("#[arcorn::rewrite] expects all types to be mangled and de-aliased.");
            match ident.to_string().as_str() {
                "i32" => "int32",
                "i64" => "int64",
                "bool" => "bool",
                "f32" => "float",
                "f64" => "double",
                "u32" => "uint32",
                "u64" => "uint64",
                "String" => "string",
                message => message,
            }
            .to_string()
        }
        _ => panic!("#[arcorn::rewrite] expects all types to be mangled and de-aliased."),
    };
    let ident = syn::Ident::new(&ty, pm2::Span::call_site());
    if let Some(tag) = tag {
        let lit = syn::LitStr::new(&format!("{}", tag), pm2::Span::call_site());
        syn::parse_quote! {
            #[prost(#ident, tag = #lit)]
        }
    } else {
        syn::parse_quote! {
            #[prost(#ident)]
        }
    }
}