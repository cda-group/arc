/// Module for translating the HIR and DFG into MLIR.
pub(crate) mod display;
/// Module for evaluating the HIR into a DFG.
pub(crate) mod from;
/// Module for evaluating the HIR into a DFG.
pub(crate) mod lower;
/// Module for representing the dataflow graph.
pub(crate) mod repr;
/// Module for verifying the dataflow graph.
pub(crate) mod verify;

pub(crate) use display::pretty;
pub(crate) use repr::*;
