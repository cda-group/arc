#![allow(unused)]

use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub enum Aggregator<F0, F1, F2, F3> {
    Monoid {
        lift: F0,
        combine: F1,
        identity: F2,
        lower: F3,
    },
}

impl<I, P, O> Aggregator<fn(I) -> P, fn(P, P) -> P, fn() -> P, fn(P) -> O> {
    pub fn monoid(
        lift: fn(I) -> P,
        combine: fn(P, P) -> P,
        identity: fn() -> P,
        lower: fn(P) -> O,
    ) -> Self {
        Self::Monoid {
            lift,
            combine,
            identity,
            lower,
        }
    }
}
