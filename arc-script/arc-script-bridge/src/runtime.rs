//! Macro implementation.

use arc_script_core::compiler::compile;
use arc_script_core::compiler::info::diags::sink::Buffer;
use arc_script_core::prelude::modes::{Input, Mode, Output};

use proc_macro as pm;
use quote::quote;

use std::fs;

/// See [`super::runtime`] for documentation.
pub(crate) fn expand(attr: pm::TokenStream, item: pm::TokenStream) -> pm::TokenStream {
    todo!()
}
