//! Macros for compiling and including arc-scripts.

#![allow(unused)]
#![feature(proc_macro_span)]
#![feature(option_expect_none)]

extern crate proc_macro;

use proc_macro as pm;

mod compile;
mod comptime;
mod include;
mod runtime;

/// Compiles an arc-script into rust-code which is placed inside a rust-module. Any items declared 
/// within the module are visible to the generated rust code code. The path of the script is 
/// relative to the file wherein the macro is used.
///
/// Caveats:
/// * `comptime` does not support staging, i.e., the compilation can only take static inputs.
///
/// ```ignore
/// #[arc_script::comptime("path/to/script/script.arc")]
/// mod script {
///     use foo;
///     use bar::baz;
/// }
/// ```
#[proc_macro_attribute]
pub fn comptime(attr: pm::TokenStream, item: pm::TokenStream) -> pm::TokenStream {
    comptime::expand(attr, item)
}

/// This macro is intended to work like `comptime` but at runtime through dynamic linking. Since 
/// Rust's ABI is not stable it cannot be implemented as of yet.
#[proc_macro_attribute]
pub fn runtime(attr: pm::TokenStream, item: pm::TokenStream) -> pm::TokenStream {
    runtime::expand(attr, item)
}

/// #[include("foo.bar")]
/// Includes an arc-script which was previously compiled with `arc_script_build::Script`.
/// The path of the script is relative to the file wherein the macro is used.
///
/// ```ignore
/// #[arc_script::include("path/to/script/script.arc")]
/// mod script {
///     use foo;
///     use bar::baz;
/// }
/// ```
#[proc_macro_attribute]
pub fn include(attr: pm::TokenStream, item: pm::TokenStream) -> pm::TokenStream {
    include::expand(attr, item)
}

/// Compiles an arc-script file into an expression which provides methods for staging the script's 
/// functions. When staged, the script can be compiled into an `.rs` file and placed in the 
/// `target/` directory.
///
/// ```ignore
/// arc_script::compile!("script.arc")
///    .stage_fun1(1, 2, 3)
///    .stage_fun2("foo", "bar")
///    .finalize()
/// ```
#[proc_macro]
pub fn compile(input: pm::TokenStream) -> pm::TokenStream {
    compile::expand(input)
}
