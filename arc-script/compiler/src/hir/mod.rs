/// Module for representing the HIR.
pub(crate) mod repr;
/// Module for lowering the AST into an HIR of definition-tables.
pub(crate) mod lower;
/// Module for converting into the HIR.
pub(crate) mod from;
/// Module for converting the HIR into a more simplified HIR.
pub(crate) mod elaborate;
/// Module for doing additional checks on the typed HIR.
pub(crate) mod check;
/// Module for inferring types of the HIR.
pub(crate) mod infer;

pub(crate) mod clone;
pub(crate) mod utils;

pub(crate) mod debug;
pub(crate) mod display;

pub(crate) use repr::*;
