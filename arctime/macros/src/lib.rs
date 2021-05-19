#![allow(unused)]

use proc_macro as pm;
use proc_macro2 as pm2;
use quote::quote;
use syn::*;

mod extract;
mod generate;

/// Generate a task component. See `tests/` for examples of how to use this macro.
#[proc_macro_attribute]
pub fn rewrite(attr: pm::TokenStream, item: pm::TokenStream) -> pm::TokenStream {
    let args = syn::parse_macro_input!(attr as syn::AttributeArgs);
    let module = syn::parse_macro_input!(item as syn::ItemMod);
    let ast = extract::extract(args, module);
    generate::generate(ast).into()
}
