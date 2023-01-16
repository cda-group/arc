use proc_macro as pm;
use quote::quote;

pub(crate) fn export(_attr: syn::AttributeArgs, item: syn::ItemImpl) -> pm::TokenStream {
    let mut functions = Vec::new();
    let ty_name = if let syn::Type::Path(x) = item.self_ty.as_ref() {
        &x.path.segments.last().unwrap().ident
    } else {
        unreachable!("Found non-path type in impl")
    };
    let (_, ty_generics, _) = item.generics.split_for_impl();
    let impl_generics = &item.generics.params;
    let where_clause = &item.generics.where_clause;
    for item in &item.items {
        if let syn::ImplItem::Method(item) = item {
            let mut inputs = item.sig.inputs.clone().into_iter().collect::<Vec<_>>();
            let method_name = &item.sig.ident;
            let name = crate::utils::new_id(format!("{}_{}", ty_name, method_name));
            let output = &item.sig.output;
            // Replace receiver with parameter
            if let Some(syn::FnArg::Receiver(r)) = inputs.first() {
                assert!(r.reference.is_none(), "Found reference to receiver");
                inputs[0] = syn::parse_quote!(self_param: #ty_name #ty_generics);
            }
            let (ids, tys): (Vec<_>, Vec<_>) = inputs
                .iter()
                .map(|i| match i {
                    syn::FnArg::Receiver(_) => unreachable!(),
                    syn::FnArg::Typed(i) => (&i.pat, &i.ty),
                })
                .unzip();
            // Merge generics and predicates
            let generics = match (&item.sig.generics.params, impl_generics) {
                (gs0, gs1) if !gs0.is_empty() && !gs1.is_empty() => quote!(<#gs0, #gs1>),
                (gs0, _) if !gs0.is_empty() => quote!(<#gs0>),
                (_, gs1) if !gs1.is_empty() => quote!(<#gs1>),
                _ => quote!(),
            };
            let where_clause = match (where_clause, &item.sig.generics.where_clause) {
                (Some(w0), Some(w1)) => quote!(where #w0, #w1),
                (Some(w), None) | (None, Some(w)) => quote!(where #w),
                (None, None) => quote!(),
            };
            // Construct wrapper
            let item = if item.sig.asyncness.is_some() {
                quote! {
                    pub async fn #name #generics ((#(#ids,)*):(#(#tys,)*)) #output #where_clause {
                        #ty_name::#method_name(#(#ids,)*).await
                    }
                }
            } else {
                quote! {
                    pub fn #name #generics (#(#ids: #tys,)*) #output #where_clause {
                        #ty_name::#method_name(#(#ids,)*)
                    }
                }
            };
            functions.push(item);
        }
    }

    quote::quote!(
        #item
        #(#functions)*
    )
    .into()
}
