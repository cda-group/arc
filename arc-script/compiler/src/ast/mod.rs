//! Meta-module of the `AST` IR.

#![allow(missing_docs)]

/// Module for debugging the `AST`.
pub(crate) mod debug;
/// Module for pretty printing the `AST`.
pub(crate) mod display;
/// Module which parses files of source code into an `AST`.
pub mod lower;
/// Module implements the from conversion.
pub mod from;
/// Data representation of the `AST`.
pub mod repr;
/// Misc functions for managing the `AST`.
pub(crate) mod utils;

/// Module for representing the `AST` data structure.
pub use repr::*;
