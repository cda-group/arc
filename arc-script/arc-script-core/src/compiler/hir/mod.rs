/// Module for representing the HIR.
pub(crate) mod repr;
/// Module for lowering the AST into an HIR of definition-tables.
pub(crate) mod from;

mod utils;

pub(crate) mod debug;
pub(crate) mod display;

pub(crate) use display::pretty;
pub(crate) use repr::*;
