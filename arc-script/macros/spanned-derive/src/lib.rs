extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use syn::{parse_macro_input, DataStruct, DeriveInput, Type};

/// Expands:
///
/// #[derive(Spanned)]
/// struct Foo { span: Span, bar: Bar }
///
/// Into:
///
/// impl From<Spanned<Bar>> for Foo {
///     fn from(Spanned(l, bar, r): Spanned<Bar>) -> Self {
///         let span = Span::new(l as u32, r as u32);
///         Self { span, bar }
///     }
/// }
#[proc_macro_derive(Spanned)]
pub fn derive_spanned(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let id = &input.ident;
    let expanded = match &input.data {
        syn::Data::Struct(data) => {
            let (ids, tys) = split_ids_tys(data);
            quote! {
                impl From<Spanned<#(#tys),*>> for #id {
                    fn from(Spanned(l, #(#ids),*, r): Spanned<#(#tys),*>) -> Self {
                        let span = Span::new(l as u32, r as u32);
                        Self { #(#ids),*, span }
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

/// Expands:
///
/// #[derive(MaybeSpanned)]
/// struct Foo { span: Span, bar: Bar }
///
/// Into:
///
/// impl From<Spanned<Bar>> for Foo {
///     fn from(Spanned(l, bar, r): Spanned<Bar>) -> Self {
///         let span = Some(Span::new(l as u32, r as u32));
///         Self { span, bar }
///     }
/// }
///
/// impl Foo {
///     fn new(bar: Bar) -> Self {
///         let span = None;
///         Self { span, bar }
///     }
/// }
#[proc_macro_derive(MaybeSpanned)]
pub fn derive_maybe_spanned(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let id = &input.ident;
    let expanded = match &input.data {
        syn::Data::Struct(data) => {
            let (ids, tys) = split_ids_tys(data);
            quote! {
                impl From<Spanned<#(#tys),*>> for #id {
                    fn from(Spanned(l, #(#ids),*, r): Spanned<#(#tys),*>) -> Self {
                        let span = Some(Span::new(l as u32, r as u32));
                        Self { #(#ids),*, span }
                    }
                }
                impl From<#(#tys),*> for #id {
                    fn from(#(#ids: #tys),*) -> Self {
                        let span = None;
                        Self { #(#ids)*, span }
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

fn split_ids_tys(data: &DataStruct) -> (Vec<&Ident>, Vec<&Type>) {
    let fields = data
        .fields
        .iter()
        .filter(|field| matches!(&field.ident, Some(id) if *id != "span"));
    let ids = fields.clone().filter_map(|field| field.ident.as_ref());
    let tys = fields.clone().map(|field| &field.ty);
    (ids.collect(), tys.collect())
}
