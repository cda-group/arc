use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use syn::{parse_macro_input, DataStruct, DeriveInput, Type};

pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let id = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let expanded = match &input.data {
        syn::Data::Struct(data) => {
            let (ids, tys, tyids) = split_ids_tys(data);
            quote! {
                impl #impl_generics From<Spanned<(#(#tys),*)>> for #id #ty_generics #where_clause {
                    fn from(Spanned(file, lhs, (#(#ids),*), rhs): Spanned<(#(#tys),*)>) -> Self {
                        Self { #(#ids),*, loc: Loc::from_range(file, lhs..rhs) }
                    }
                }
                impl #impl_generics #id #ty_generics #where_clause {
                    pub(crate) fn syn(#(#tyids),*) -> Self {
                        Self { #(#ids),*, loc: Loc::Fake }
                    }
                }
            }
        }
        _ => quote! {
            compile_error!("Expected a struct.")
        },
    };
    TokenStream::from(expanded)
}

fn split_ids_tys(data: &DataStruct) -> (Vec<&Ident>, Vec<&Type>, Vec<proc_macro2::TokenStream>) {
    let fields = data
        .fields
        .iter()
        .filter(|field| matches!(&field.ident, Some(id) if *id != "loc"));
    let ids = fields.clone().map(|field| field.ident.as_ref().unwrap());
    let tys = fields.clone().map(|field| &field.ty);
    let tyids = fields.clone().map(|field| {
        let id = &field.ident;
        let ty = &field.ty;
        quote!(#id : #ty)
    });
    (ids.collect(), tys.collect(), tyids.collect())
}
