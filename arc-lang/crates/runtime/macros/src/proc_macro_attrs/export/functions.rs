use proc_macro as pm;
use quote::quote;

pub(crate) fn export(_attr: syn::AttributeArgs, item: syn::ItemFn) -> pm::TokenStream {
    let mut inputs = item.sig.inputs.clone().into_iter().collect::<Vec<_>>();
    let fn_name = &item.sig.ident;
    let name = crate::utils::new_id(format!("export_{}", fn_name));
    let output = &item.sig.output;
    let (impl_generics, _, where_clause) = item.sig.generics.split_for_impl();
    let (ids, tys): (Vec<_>, Vec<_>) = inputs
        .iter()
        .map(|i| match i {
            syn::FnArg::Receiver(_) => unreachable!(),
            syn::FnArg::Typed(i) => (&i.pat, &i.ty),
        })
        .unzip();
    if item.sig.asyncness.is_some() {
        quote! {
            #item
            pub async fn #name #impl_generics ((#(#ids,)*):(#(#tys,)*)) #output #where_clause {
                #fn_name(#(#ids,)*).await
            }
        }
    } else {
        quote! {
            #item
            pub fn #name #impl_generics ((#(#ids,)*):(#(#tys,)*)) #output #where_clause {
                #fn_name(#(#ids,)*)
            }
        }
    }
    .into()
}
