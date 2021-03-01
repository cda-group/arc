use proc_macro::TokenStream;

mod rewrite;

/// Declares a new enum which is compatible with the `arcorn::{enwrap, unwrap, is}` API.
///
/// Any expansion of the macro satisfies the following properties:
/// * Enums:
///   * Each enum is wrapped as an `Option` inside a struct (prost requirement).
///   * Each enum implements a method `.wrap()` to wrap it inside the struct.
///   * Each enum variants is imported into the global namespace.
///
/// * Structs
#[proc_macro_attribute]
pub fn rewrite(_: TokenStream, input: TokenStream) -> TokenStream {
    rewrite::execute(input)
}
