#![allow(unused)]

use arc_runtime::prelude::*;

declare!(
    functions: [identity_i32, identity_i64],
    tasks: []
);

#[rewrite(unmangled = "identity")]
fn identity_i32(x: i32) -> i32 {}

#[rewrite(unmangled = "identity")]
fn identity_i64(x: i64) -> i64 {}

// TODO: `i128` and `u128` are not yet supported by `serde_state`
// #[rewrite(unmangled = "identity")]
// fn identity_i128(x: i128) -> i128 {}

#[rewrite]
fn identity<T>(x: T) -> T {
    x
}
