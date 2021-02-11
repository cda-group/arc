use arc_script_core::compiler::compile;
use arc_script_core::compiler::info::diags::sink::Buffer;
use arc_script_core::prelude::modes::{Input, Mode, Output};

use proc_macro as pm;
use quote::quote;

use std::fs;
use std::path::PathBuf;

/// arc_script::compile!("script.arc")
///    .stage_fun1(1, 2, 3)
///    .stage_fun2("foo", "bar")
///    .finalize()
pub(crate) fn expand(input: pm::TokenStream) -> pm::TokenStream {
    todo!()
}
