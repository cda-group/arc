pub(crate) use super::Context;
/// Module for lowering call-expressions.
pub(crate) mod call;
/// Module for lowering call-expressions.
//     pub(crate) mod cases;
/// Module for lowering if-let-expressions patterns.
pub(crate) mod if_else;
/// Module for lowering pure lambdas into first-class functions.
pub(crate) mod lambda;
/// Module for lowering let-expressions patterns.
pub(crate) mod let_in;
/// Module for lifting expressions into functions.
pub(crate) mod lift;
/// Module for lowering path expressions.
pub(crate) mod path;
/// Module for lowering patterns.
pub(crate) mod pattern;
/// Module for lowering binop-expressions.
pub(crate) mod ops;
/// Debug utils.
pub(crate) mod debug;
pub(crate) mod task;
pub(crate) mod fsm;
pub(crate) mod loops;
