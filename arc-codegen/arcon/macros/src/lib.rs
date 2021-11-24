use proc_macro::TokenStream;

#[cfg(feature = "backend_arcon")]
mod prost;
mod tasks;
mod structs;
mod enums;

/// Declares a new enum which is compatible with the `codegen::{enwrap, unwrap, is}` API.
///
/// Any expansion of the macro satisfies the following properties:
/// * Enums:
///   * Each enum is wrapped as an `Option` inside a struct (prost requirement).
///   * Each enum implements a method `.wrap()` to wrap it inside the struct.
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
