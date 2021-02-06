#![allow(unused)]
#![deny(clippy::all)]
// #![deny(warnings)]
#![deny(unsafe_code)]
// #![deny(missing_docs)]

#[macro_use]
extern crate shrinkwraprs;
#[macro_use]
extern crate educe;
#[macro_use]
extern crate derive_more;

/// Module which assembles the compilation pipeline.
pub mod compiler;
/// Module which re-exports common functionality.
pub mod prelude;
