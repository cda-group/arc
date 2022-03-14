use proc_macro as pm;
use quote::quote;

pub(crate) fn rewrite(_attr: syn::AttributeArgs, item: syn::ItemImpl) -> pm::TokenStream {
    use crate::new_id;
    let mut functions = Vec::new();
    let ty_name = if let syn::Type::Path(x) = item.self_ty.as_ref() {
        &x.path.segments.last().unwrap().ident
    } else {
        unreachable!("Found non-path type in impl")
    };
    let (impl_generics, ty_generics, where_clause) = item.generics.split_for_impl();
    for item in &item.items {
        if let syn::ImplItem::Method(item) = item {
            let mut inputs = item.sig.inputs.clone().into_iter().collect::<Vec<_>>();
            let method_name = &item.sig.ident;
            let name = new_id(format!("{}_{}", ty_name, method_name));
            let output = &item.sig.output;
            if matches!(inputs[0], syn::FnArg::Receiver(_)) {
                inputs[0] = syn::parse_quote!(self_param: #ty_name #ty_generics);
            }
            let ctx = match inputs.pop().unwrap() {
                syn::FnArg::Receiver(_) => unreachable!("Receiver in impl method"),
                syn::FnArg::Typed(p) => p,
            };
            let ctx_id = ctx.pat;
            let ctx_ty = ctx.ty;
            let (ids, tys): (Vec<_>, Vec<_>) = inputs
                .iter()
                .map(|i| match i {
                    syn::FnArg::Receiver(_) => unreachable!(),
                    syn::FnArg::Typed(i) => (&i.pat, &i.ty),
                })
                .unzip();
            functions.push(quote! {
                pub fn #name #impl_generics ((#(#ids,)*):(#(#tys,)*), #ctx_id: #ctx_ty) #output #where_clause {
                    #ty_name::#method_name(#(#ids,)* #ctx_id)
                }
            });
        }
    }

    quote::quote!(
        #item
        #(#functions)*
    )
    .into()
}
