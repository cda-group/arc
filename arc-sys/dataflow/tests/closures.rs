use dataflow::prelude::*;

#[rewrite]
fn f(a: i32) -> i32 {
    a + a
}

#[test]
fn test() {
    let x0: fn(i32) -> i32 = f;
    let x1: fn(i32) -> i32 = f;
    x0(1);
    x1(1);
    f(1);
    f(1);
}
