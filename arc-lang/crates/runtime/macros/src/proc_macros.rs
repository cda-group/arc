use proc_macro::TokenStream;

pub fn unwrap(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let expr: syn::Expr = crate::utils::parse(&mut iter);
    let path: syn::Path = crate::utils::parse(&mut iter);
    quote::quote!(if let #path(v) = &#expr { v.clone() } else { unreachable!() }).into()
}
