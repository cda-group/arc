#![allow(unused)]

use dataflow::prelude::*;

#[rewrite]
fn plus_one(x: i32) -> i32 {
    x + 1
}

#[rewrite]
fn main() {
    plus_one(1);
}
