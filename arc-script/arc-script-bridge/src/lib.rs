#![allow(unused)]
#![feature(proc_macro_span)]
#![feature(option_expect_none)]

extern crate proc_macro;

use proc_macro as pm;

mod compile;
mod comptime;
mod include;
mod runtime;

/// #[arc_script("lib.arc")]
/// mod script;
#[proc_macro_attribute]
pub fn arc_script(attr: pm::TokenStream, item: pm::TokenStream) -> pm::TokenStream {
    comptime::expand(attr, item)
}

/// #[arc_script_dyn("lib.arc")]
/// mod script;
#[proc_macro_attribute]
pub fn arc_script_dyn(attr: pm::TokenStream, item: pm::TokenStream) -> pm::TokenStream {
    runtime::expand(attr, item)
}

/// #[include("foo.bar")]
#[proc_macro_attribute]
pub fn include(attr: pm::TokenStream, item: pm::TokenStream) -> pm::TokenStream {
    include::expand(attr, item)
}

/// #[include("foo.bar")]
#[proc_macro]
pub fn compile(input: pm::TokenStream) -> pm::TokenStream {
    compile::expand(input)
}
