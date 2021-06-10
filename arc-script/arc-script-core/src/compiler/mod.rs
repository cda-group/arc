/// Module for representing Arc Queries.
#[cfg(feature = "query")]
pub(crate) mod query;
/// Module for representing the Abstract Syntax Tree.
///
/// NB: This module is public so that external libraries can construct ASTs
/// without having to generate source code.
pub mod ast;
/// Module for representing Higher Order Intermediate Representation code.
pub(crate) mod hir;
/// Module for representing side information.
pub mod info;
/// Module for representing Multi-Level Intermediate Representation code.
pub(crate) mod mlir;
/// Module for representing Arcorn code.
pub(crate) mod arcorn;

/// Module which assembles compilation-pipeline.
pub mod pipeline;

pub use pipeline::compile;
