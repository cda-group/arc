use arc_runtime::prelude::*;

declare_functions!(x);

// #[rewrite(nonpersistent)]
// mod source {
//     fn task(i: Vec<i32>, #[output] o: Pushable<i32>) {
//         for x in i.into_iter().cloned() {
//             push!(o, x);
//         }
//     }
// }

#[rewrite(persistent)]
mod map {
    fn task(a: Pullable<i32>, f: function!((i32) -> i32), #[output] b: Pushable<i32>) {}

    struct State0 {
        a: Pullable<i32>,
        f: function!((i32) -> i32),
        b: Pushable<i32>,
    }

    struct State1 {
        a: Pullable<i32>,
        f: function!((i32) -> i32),
        b: Pushable<i32>,
        pull: BoxFuture<'static, Control<i32>>,
    }

    struct State2 {
        a: Pullable<i32>,
        f: function!((i32) -> i32),
        b: Pushable<i32>,
        push: BoxFuture<'static, Control<()>>,
    }

    struct State3 {}

    enum State {
        State0(State0),
        State1(State1),
        State2(State2),
        State3(State3),
    }

    fn transition0(
        State0 {
            mut a,
            mut b,
            mut f,
        }: State0,
        _cx: &mut PollContext,
        ctx: Context,
    ) -> (Poll<()>, State) {
        pull_transition!(pull, a, State1 { a, b, f, pull });
    }

    fn transition1(
        State1 {
            mut a,
            mut b,
            mut f,
            mut pull,
        }: State1,
        cx: &mut PollContext,
        ctx: Context,
    ) -> (Poll<()>, State) {
        let x = wait!(pull, cx, State1 { a, b, f, pull }, State3 {});
        let y = call_indirect!(f(x));
        push_transition!(push, b, y, State2 { a, b, f, push });
    }

    fn transition2(
        State2 {
            mut a,
            mut b,
            mut f,
            mut push,
        }: State2,
        cx: &mut PollContext,
        ctx: Context,
    ) -> (Poll<()>, State) {
        wait!(push, cx, State2 { a, b, f, push }, State0 { a, b, f });
        transition!(State0 { a, b, f });
    }

    fn transition3(State3 {}: State3, _cx: &mut PollContext, ctx: Context) -> (Poll<()>, State) {
        unreachable!()
    }
}

// #[rewrite(nonpersistent)]
// mod log {
//     fn task(i: Pullable<i32>) {
//         loop {
//             println!("Logging {}", pull!(i));
//         }
//     }
// }

#[rewrite]
fn x(x: i32) -> i32 {
    x + 1
}

// #[rewrite(main)]
// #[test]
// fn rewrite_impersistent_task() {
//     let v = vector!(1i32, 2, 3);
//     let s = direct_call!(source(*v));
//     let s = direct_call!(map(*s, function!(x)));
//     let s = direct_call!(log(*s));
// }
