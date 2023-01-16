use num::Zero;

use crate::data::Data;

// Requires associativity, and invertibility.
pub trait Aggregator {
    type I: Data;
    type P: Data;
    type O: Data;

    fn lift(i: Self::I) -> Self::P;
    fn merge(a: Self::P, b: Self::P) -> Self::P;
    fn identity() -> Self::P;
    fn lower(p: Self::P) -> Self::O;
}
