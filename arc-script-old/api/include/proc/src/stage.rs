//! Macro implementation.

use arc_script_compiler::compile;
use arc_script_compiler::info::diags::sink::Buffer;
use arc_script_compiler::prelude::modes::{Input, Mode, Output};

use proc_macro as pm;
use quote::quote;

use std::fs;
use std::path::PathBuf;

/// See [`super::compile`] for documentation.
pub(crate) fn expand(input: pm::TokenStream) -> pm::TokenStream {
    todo!()
}
