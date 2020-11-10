#![allow(unused)]
#![feature(proc_macro_span)]
#![feature(option_expect_none)]

extern crate proc_macro;

use arc_script_core::compiler::compile;
use arc_script_core::compiler::info::diags::sink::Buffer;
use arc_script_core::prelude::modes::{Input, Mode, Output};

use proc_macro as pm;
use quote::quote;

use std::fs;

mod comptime;
mod runtime;

/// #[arc_script("lib.arc")]
/// mod script;
#[proc_macro_attribute]
pub fn arc_script(attr: pm::TokenStream, item: pm::TokenStream) -> pm::TokenStream {
    comptime::arc_script(attr, item)
}

/// #[arc_script_dyn("lib.arc")]
/// mod script;
#[proc_macro_attribute]
pub fn arc_script_dyn(attr: pm::TokenStream, item: pm::TokenStream) -> pm::TokenStream {
    runtime::arc_script(attr, item)
}
