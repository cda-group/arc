#![allow(dead_code)]
#![allow(clippy::let_unit_value)]

use dataflow::prelude::*;

#[rewrite]
fn foo(x: i32) {
    let a: i32 = x - 1;
    if x == 0 {
        println!("Hello, world!");
    } else {
        call!(foo(a))
    }
}

fn main() {
    let x: Str = call!(Str_from_str("Hello, world!"));
    let y: &str = "Hello, world!";
    let _z: unit = call!(Str_push_str(x, y));
}
