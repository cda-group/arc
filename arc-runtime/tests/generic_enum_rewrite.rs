#![allow(unused)]

use arc_runtime::prelude::*;

#[rewrite]
pub enum Foo<A> {
    X(A),
}

#[rewrite]
pub enum Bar<A, B> {
    X(A),
    Y(B),
}

type Baz = Bar<Foo<i32>, Bar<i32, i32>>;
