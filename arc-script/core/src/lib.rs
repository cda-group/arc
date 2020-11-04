#![deny(clippy::all)]

pub use anyhow::Result;

#[macro_use]
extern crate educe;

mod codegen;
mod repr;
mod cli;
mod passes;
pub mod prelude;
pub mod compiler;
