use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DataStruct, DeriveInput};

pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let id = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let expanded = match &input.data {
        syn::Data::Struct(data) => {
            let t = get_type(data);
            quote! {
                impl #impl_generics From<#id> for #t #ty_generics #where_clause {
                    fn from(data: #id) -> Self {
                        data.id
                    }
                }
                impl<'i> From<&'i #id> for #t {
                    fn from(data: &'i #id) -> Self {
                        data.id
                    }
                }
                impl<'i> From<&'i #t> for #t {
                    fn from(id: &'i #t) -> Self {
                        *id
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

fn get_type(data: &DataStruct) -> &syn::Type {
    &data
        .fields
        .iter()
        .find(|field| matches!(&field.ident, Some(id) if *id == "id"))
        .unwrap()
        .ty
}
