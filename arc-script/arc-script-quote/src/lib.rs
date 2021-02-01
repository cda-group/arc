#![feature(proc_macro_span)]

//pub(crate) mod lexer;
pub(crate) mod string;

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro]
pub fn arc_script(input: TokenStream) -> TokenStream {
    string::compile_literal(input);
    TokenStream::new()
}
