#![allow(unused)]

use dataflow::prelude::*;

#[rewrite(unmangled = "identity")]
fn identity_i32(x: i32) -> i32 {}

#[rewrite(unmangled = "identity")]
fn identity_i64(x: i64) -> i64 {}

#[rewrite(unmangled = "Foo::identity")]
fn identity_foo(x: i32) -> i32 {}

// TODO: `i128` and `u128` are not yet supported by `serde_state`
// #[rewrite(unmangled = "identity")]
// fn identity_i128(x: i128) -> i128 {}

#[rewrite]
fn identity<T>(x: T) -> T {
    x
}

struct Foo {}

impl Foo {
    #[rewrite]
    fn identity(x: i32) -> i32 {
        x
    }
}
