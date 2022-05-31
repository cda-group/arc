#![allow(unused)]

use arc_runtime::prelude::*;

declare! {
    functions: [x],
    tasks: [
        map_impl(MapState)
    ]
}

#[rewrite]
fn map(i: PullChan<i32>, f: function!((i32) -> i32)) -> PullChan<i32> {
    let (w, r) = call!(channel());
    let state = call!(map_init(i, f, w));
    spawn!(state);
    r
}

#[rewrite]
struct State0 {
    i: PullChan<i32>,
    f: function!((i32) -> i32),
    o: PushChan<i32>,
}

#[rewrite]
enum MapState {
    State0(State0),
}

#[rewrite]
fn map_init(i: PullChan<i32>, f: function!((i32) -> i32), o: PushChan<i32>) -> MapState {
    enwrap!(MapState::State0, new!(State0 { i, f, o }))
}

#[rewrite]
async fn map_impl(state: MapState) {
    loop {
        if is!(MapState::State0, state) {
            let s0 = unwrap!(MapState::State0, state);
            let i = s0.i;
            let f = s0.f;
            let o = s0.o;
            let x = call_async!(PullChan_pull(i));
            let y = call_indirect!(f(x));
            call_async!(PushChan_push(o, y));
        }
    }
}

#[rewrite]
fn x(x: i32) -> i32 {
    x + 1
}

#[rewrite(main)]
#[test]
fn main() {
    // let v = vector!(1i32, 2, 3);
    // let s = call!(source(v));
    // let s = call!(map(s, function!(x)));
    // call!(log(s));
}
