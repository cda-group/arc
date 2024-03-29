use proc_macro as pm;
use quote::quote;

use crate::new_id;

#[allow(unused)]
pub(crate) fn rewrite(args: syn::AttributeArgs, mut item: syn::ItemStruct) -> pm::TokenStream {
    item.fields.iter_mut().for_each(|field| {
        field.vis = syn::parse_quote!(pub);
    });
    let (_, type_generics, where_clause) = item.generics.split_for_impl();

    let mut sharable_impl_generics = item.generics.params.clone();
    let mut sendable_impl_generics = item.generics.params.clone();

    sharable_impl_generics.iter_mut().for_each(|g| {
        if let syn::GenericParam::Type(t) = g {
            t.bounds.push(syn::parse_quote!(Sharable))
        }
    });
    sendable_impl_generics.iter_mut().for_each(|g| {
        if let syn::GenericParam::Type(t) = g {
            t.bounds.push(syn::parse_quote!(Sendable))
        }
    });
    let sharable_impl_generics = quote::quote!(<#sharable_impl_generics>);
    let sendable_impl_generics = quote::quote!(<#sendable_impl_generics>);

    let into_generics: Vec<_> = item
        .generics
        .params
        .iter()
        .filter_map(|g| {
            if let syn::GenericParam::Type(t) = g {
                let id = &t.ident;
                Some(quote::quote!(#id::T))
            } else {
                None
            }
        })
        .collect();
    let into_generics = if into_generics.is_empty() {
        quote::quote!()
    } else {
        quote::quote!(<#(#into_generics),*>)
    };

    let abstract_id = item.ident.clone();
    let concrete_id = new_id(format!("Concrete{}", item.ident));
    let sharable_mod_id = new_id(format!("sharable_{}", item.ident));
    let sendable_mod_id = new_id(format!("sendable_{}", item.ident));

    let mut concrete_sharable_item = item.clone();
    let mut concrete_sendable_item = item.clone();

    concrete_sharable_item
        .generics
        .params
        .iter_mut()
        .for_each(|g| {
            if let syn::GenericParam::Type(t) = g {
                t.bounds.push(syn::parse_quote!(Sharable))
            }
        });
    concrete_sendable_item
        .generics
        .params
        .iter_mut()
        .for_each(|g| {
            if let syn::GenericParam::Type(t) = g {
                t.bounds.push(syn::parse_quote!(Sendable))
            }
        });

    concrete_sharable_item.ident = concrete_id.clone();
    concrete_sendable_item.ident = concrete_id.clone();

    // Generate the sendable struct
    concrete_sendable_item
        .fields
        .iter_mut()
        .for_each(|f| {
            let ty = f.ty.clone();
            if let syn::Type::Path(t) = &ty {
                if !item.generics.params.iter().any(|x| match x {
                    syn::GenericParam::Type(x) => t.path.is_ident(&x.ident),
                    _ => false,
                }) {
                    f.ty = syn::parse_quote!(<#ty as DynSharable>::T);
                }
            } else {
                f.ty = syn::parse_quote!(<#ty as DynSharable>::T);
            }
        });

    let field_id = concrete_sendable_item
        .fields
        .iter()
        .map(|f| &f.ident)
        .collect::<Vec<_>>();

    quote!(

        use arc_runtime::prelude::*;
        pub mod #sharable_mod_id {
            use super::*;
            use arc_runtime::prelude::*;

            #[derive(Clone, Debug, Send, Sync, Alloc, Unpin, From, Deref, Abstract, Collectable, Finalize, Trace)]
            pub struct #abstract_id #sharable_impl_generics(pub Gc<#concrete_id #type_generics>) #where_clause;

            #[derive(Clone, Debug, Collectable, Finalize, Trace)]
            #concrete_sharable_item
        }

        mod #sendable_mod_id {
            use super::*;
            use arc_runtime::prelude::*;

            #[derive(Clone, Debug, Deref, From, Abstract, Deserialize, Serialize)]
            #[serde(bound = "")]
            #[from(forward)]
            pub struct #abstract_id #sendable_impl_generics(pub Box<#concrete_id #type_generics>) #where_clause;

            #[derive(Clone, Debug, Deserialize, Serialize)]
            #[serde(bound = "")]
            #concrete_sendable_item
        }

        use #sharable_mod_id::#abstract_id;
        use #sharable_mod_id::#concrete_id;

        impl #sharable_impl_generics DynSharable for #sharable_mod_id::#abstract_id #type_generics #where_clause {
            type T = #sendable_mod_id::#abstract_id #into_generics;
            fn into_sendable(&self, ctx: Context) -> Self::T {
                #sendable_mod_id::#concrete_id {
                    #(#field_id: (self.0).#field_id.clone().into_sendable(ctx)),*
                }.into()
            }
        }

        impl #sendable_impl_generics DynSendable for #sendable_mod_id::#abstract_id #type_generics #where_clause {
            type T = #sharable_mod_id::#abstract_id #into_generics;
            fn into_sharable(&self, ctx: Context) -> Self::T {
                #sharable_mod_id::#concrete_id {
                    #(#field_id: (self.0).#field_id.into_sharable(ctx)),*
                }.alloc(ctx)
            }
        }

    ).into()
}
