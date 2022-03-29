#![allow(unused)]

use arc_runtime::prelude::*;

#[rewrite]
pub struct Foo<A> {
    pub a: A,
}

#[rewrite]
pub struct Bar<A, B> {
    pub a: A,
    pub b: B,
}

type Baz = Bar<Foo<i32>, Bar<i32, i32>>;
