use arc_runtime::prelude::*;

declare_functions!(f);

#[rewrite]
fn f(x: i32) -> i32 {
    x + 1
}

#[rewrite(nonpersistent)]
async fn source(mut i: Vec<i32>, #[output] mut o: Pushable<i32>) {
    for x in i.into_iter().cloned() {
        push!(o, x);
    }
}

#[rewrite(nonpersistent)]
async fn map(mut i: Pullable<i32>, mut f: function!((i32) -> i32), #[output] mut o: Pushable<i32>) {
    loop {
        let x = pull!(i);
        let y = call_indirect!(f(x));
        push!(o, y);
    }
}

#[rewrite(nonpersistent)]
async fn log(mut i: Pullable<i32>) {
    loop {
        println!("Logging {}", pull!(i));
    }
}

use arc_runtime::data::channels::local::multicast::Pullable;

#[rewrite(main)]
#[test]
fn rewrite_impersistent_task() {
    let v: Vec<i32> = vector![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let s: Pullable<i32> = call!(source(v));
    let f: function!((i32) -> i32) = function!(f);
    let s: Pullable<i32> = call!(map(s, f));
    call!(log(s));
}
