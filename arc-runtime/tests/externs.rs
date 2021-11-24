#![allow(unused)]

use arc_runtime::prelude::*;

#[rewrite(unmangled = "add")]
fn addi32(x: i32) -> i32 {}

#[rewrite(unmangled = "add")]
fn addi64(x: i64) -> i64 {}

#[rewrite(unmangled = "add")]
fn addi128(x: i128) -> i128 {}

fn add<T>(x: T) -> T {
    x
}
