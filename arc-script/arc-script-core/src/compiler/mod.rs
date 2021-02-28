/// Module for representing Arc Queries.
pub(crate) mod query;
/// Module for representing the Abstract Syntax Tree.
///
/// NB: This module is public so that external libraries can construct ASTs
/// without having to generate source code.
pub mod ast;
/// Module for representing Dataflow graphs. Exports data types for staging
/// `HIR` functions.
pub mod dfg;
/// Module for representing Higher Order Intermediate Representation code.
pub(crate) mod hir;
/// Module for representing side information.
pub mod info;
/// Module for representing Multi-Level Intermediate Representation code.
pub(crate) mod mlir;
/// Module for representing Arcon code.
pub(crate) mod arcon;

// Module for incremental compilation.
// pub(crate) mod database;
/// Module which assembles compilation-pipeline.
pub mod pipeline;

pub(crate) mod pretty;

pub use pipeline::compile;
