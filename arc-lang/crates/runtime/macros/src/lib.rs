use proc_macro::TokenStream;

mod proc_macro_attrs;
mod proc_macro_derives;
mod proc_macros;
pub(crate) mod utils;

#[proc_macro_derive(Send)]
pub fn derive_send(input: TokenStream) -> TokenStream {
    proc_macro_derives::derive_send(syn::parse_macro_input!(input as syn::DeriveInput))
}

#[proc_macro_derive(Sync)]
pub fn derive_sync(input: TokenStream) -> TokenStream {
    proc_macro_derives::derive_sync(syn::parse_macro_input!(input as syn::DeriveInput))
}

#[proc_macro_derive(Unpin)]
pub fn derive_unpin(input: TokenStream) -> TokenStream {
    proc_macro_derives::derive_unpin(syn::parse_macro_input!(input as syn::DeriveInput))
}

#[proc_macro_derive(DeepClone)]
pub fn derive_deep_clone(input: TokenStream) -> TokenStream {
    proc_macro_derives::derive_deep_clone(syn::parse_macro_input!(input as syn::DeriveInput))
}

/// Unwraps a value out of an enum-variant. Panics if it's the wrong variant.
#[proc_macro]
pub fn unwrap(input: TokenStream) -> TokenStream {
    proc_macros::unwrap(input)
}

#[proc_macro_attribute]
pub fn data(_attr: TokenStream, input: TokenStream) -> TokenStream {
    proc_macro_attrs::data(syn::parse_macro_input!(input as syn::DeriveInput))
}
