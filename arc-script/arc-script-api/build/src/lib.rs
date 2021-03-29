//! Library for compiling arc-scripts inside `build.rs` files.

#![allow(unused)]

mod build;

#[derive(Default)]
pub struct Builder {
    no_exclude: bool,
    source_dirs: Vec<String>,
}

impl Builder {
    /// Compiles all scripts, regardless of whether they are used or not.
    pub fn no_exclude(self, no_exclude: bool) -> Self {
        Self { no_exclude, ..self }
    }
    /// Start compiling from these directories. By default, compilation starts from the
    /// build-script's directory.
    pub fn source_dirs<'i>(self, source_dirs: impl AsRef<[&'i str]>) -> Self {
        Self {
            source_dirs: source_dirs.as_ref().iter().map(|x| x.to_string()).collect(),
            ..self
        }
    }
}
