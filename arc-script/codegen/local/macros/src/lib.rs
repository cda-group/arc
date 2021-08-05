use proc_macro::TokenStream;
use proc_macro2 as pm2;

mod tasks;
mod structs;
mod enums;

/// Declares a new enum which is compatible with the `codegen::{enwrap, unwrap, is}` API.
///
/// Any expansion of the macro satisfies the following properties:
/// * Enums:
///   * Each enum variants is imported into the global namespace.
/// * Structs
/// * Tasks
#[proc_macro_attribute]
pub fn rewrite(attr: TokenStream, input: TokenStream) -> TokenStream {
    let item = syn::parse_macro_input!(input as syn::Item);
    let attr = syn::parse_macro_input!(attr as syn::AttributeArgs);
    match item {
        syn::Item::Enum(item) => enums::rewrite(attr, item),
        syn::Item::Struct(item) => structs::rewrite(attr, item),
        syn::Item::Mod(item) => tasks::rewrite(attr, item),
        _ => panic!("#[codegen::rewrite] expects enum or struct as input"),
    }
}

pub(crate) fn new_id(s: impl ToString) -> syn::Ident {
    syn::Ident::new(&s.to_string(), pm2::Span::call_site())
}
