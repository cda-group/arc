use proc_macro as pm;
use quote::quote;

use crate::get_metas;
use crate::has_meta_key;
use crate::new_id;

#[allow(unused)]
pub(crate) fn rewrite(attr: syn::AttributeArgs, mut item: syn::ItemStruct) -> pm::TokenStream {
    item.vis = syn::parse_quote!(pub);
    item.fields.iter_mut().for_each(|field| {
        field.vis = syn::parse_quote!(pub);
    });
    item.generics.params.iter_mut().for_each(|g| {
        if let syn::GenericParam::Type(t) = g {
            t.bounds.push(syn::parse_quote!(Data))
        }
    });
    let (impl_generics, type_generics, where_clause) = item.generics.split_for_impl();

    let wrapper_id = item.ident.clone();
    item.ident = new_id(format!("Inner{}", wrapper_id));
    let id = &item.ident;

    let field_id = item.fields.iter().map(|f| &f.ident).collect::<Vec<_>>();

    let (inner_type, inner_expr) = if has_meta_key("compact", &get_metas(&attr)) {
        (quote::quote!(#id #type_generics), quote::quote!(data))
    } else {
        (
            quote::quote!(Gc<#id #type_generics>),
            quote::quote!(ctx.heap.allocate(data)),
        )
    };

    quote!(
        use arc_runtime::prelude::*;

        #[derive(Copy, Clone, Debug, Send, Sync, Unpin, From, Deref, Trace)]
        #[serde_state]
        pub struct #wrapper_id #impl_generics(pub #inner_type) #where_clause;

        impl #impl_generics #wrapper_id #type_generics #where_clause {
            fn new(data: #id #type_generics, mut ctx: Context<impl Execute>) -> Self {
                Self(#inner_expr)
            }
        }

        #[derive(Copy, Clone, Debug, Send, Sync, Unpin, Trace)]
        #[serde_state]
        #item
    )
    .into()
}
