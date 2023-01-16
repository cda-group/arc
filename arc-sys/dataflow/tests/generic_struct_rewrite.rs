#![allow(unused)]

use dataflow::prelude::*;

#[rewrite]
struct Foo<A> {
    a: A,
}

#[rewrite]
struct Bar<A, B> {
    a: A,
    b: B,
}

type Baz = Bar<Foo<i32>, Bar<i32, i32>>;
