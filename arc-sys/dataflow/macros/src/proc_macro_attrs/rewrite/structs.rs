use proc_macro as pm;
use quote::quote;

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
    item.ident = crate::utils::new_id(format!("_{}", wrapper_id));
    let id = &item.ident;

    let field_id = item.fields.iter().map(|f| &f.ident).collect::<Vec<_>>();

    let (inner_type, inner_expr) =
        if crate::utils::has_meta_key("compact", &crate::utils::get_metas(&attr)) {
            (quote::quote!(#id #type_generics), quote::quote!(data))
        } else {
            (
                quote::quote!(Rc<#id #type_generics>),
                quote::quote!(Rc::new(data)),
            )
        };

    quote!(
        use dataflow::prelude::*;

        #[derive(Clone, Debug, Unpin, From, Deref, Serialize, Deserialize)]
        #[serde(bound = "")]
        pub struct #wrapper_id #impl_generics(pub #inner_type) #where_clause;

        impl #impl_generics #wrapper_id #type_generics #where_clause {
            pub fn new(data: #id #type_generics) -> Self {
                Self(#inner_expr)
            }
        }

        #[derive(Clone, Debug, Unpin, Serialize, Deserialize)]
        #[serde(bound = "")]
        #item
    )
    .into()
}
