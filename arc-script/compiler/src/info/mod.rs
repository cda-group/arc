/// Module for representing and interning diagnostics.
/// NB: Public because it is used by LSP.
pub mod diags;
/// Module for interning files.
pub mod files;
/// Module for representing modes of compilation.
pub mod modes;
/// Module for interning names.
pub mod names;
/// Module for interning paths.
pub mod paths;
/// Module for interning and unifying types.
pub(crate) mod types;

/// Module for debugging info.
pub(crate) mod debug;

/// Module for representing info.
pub mod repr;

pub use repr::*;
