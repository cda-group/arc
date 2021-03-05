//! Macro implementation.

use proc_macro as pm;

use std::fs;
use std::path::{Path, PathBuf};

/// See [`super::import`] for documentation.
pub(crate) fn expand(input: pm::TokenStream) -> pm::TokenStream {
    fs::read_to_string(
        std::env::var("OUT_DIR")
            .unwrap()
            .parse::<PathBuf>()
            .unwrap()
            .join(pm::Span::call_site().source_file().path()),
    )
    .unwrap()
    .parse()
    .unwrap()
}
