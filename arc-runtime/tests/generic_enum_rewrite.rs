#![allow(unused)]

use arc_runtime::prelude::*;

#[rewrite]
enum Foo<A> {
    X(A),
}

#[rewrite]
enum Bar<A, B> {
    X(A),
    Y(B),
}

type Baz = Bar<Foo<i32>, Bar<i32, i32>>;
