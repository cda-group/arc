// #![allow(unused)]
#![feature(never_type)]
#![allow(unused)]

use arc_runtime::prelude::*;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::Context as PollContext;
use std::task::Poll;

use futures::future::BoxFuture;
use futures::FutureExt;

use arc_runtime::data::channels::local::multicast::channel;
use arc_runtime::data::channels::local::multicast::Pullable;
use arc_runtime::data::channels::local::multicast::Pushable;

enum State {
    State0 {
        a: Pullable<i32>,
        b: Pushable<i32>,
    },
    State1 {
        a: Pullable<i32>,
        b: Pushable<i32>,
        pull: BoxFuture<'static, Control<i32>>,
    },
    State3 {
        a: Pullable<i32>,
        b: Pushable<i32>,
        push: BoxFuture<'static, Control<()>>,
    },
}

impl Future for State {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut PollContext<'_>) -> Poll<Self::Output> {
        let ctx = todo!();
        replace_with::replace_with_or_abort_and_return(self.get_mut(), |state| state.transition(cx, ctx))
    }
}

impl State {
    fn transition(self, cx: &mut PollContext<'_>, ctx: Context) -> (Poll<()>, Self) {
        match self {
            State::State0 { mut a, b } => {
                let pull = a.pull(ctx).boxed();
                let pull = unsafe { std::mem::transmute(pull) };
                (Poll::Pending, State::State1 { a, b, pull })
            }
            State::State1 { a, b, mut pull } => {
                if let Poll::Ready(Control::Continue(x)) = pull.as_mut().poll(cx) {
                    let push = b.push(x, ctx).boxed();
                    let push = unsafe { std::mem::transmute(push) };
                    (Poll::Pending, State::State3 { a, b, push })
                } else {
                    (Poll::Pending, State::State1 { a, b, pull })
                }
            }
            State::State3 { a, b, mut push } => {
                if let Poll::Ready(Control::Continue(())) = push.as_mut().poll(cx) {
                    (Poll::Pending, State::State0 { a, b })
                } else {
                    (Poll::Pending, State::State3 { a, b, push })
                }
            }
        }
    }
}

#[derive(ComponentDefinition)]
struct DoThing {
    ctx: ComponentContext<Self>,
    a: Pullable<i32>,
    b: Pushable<i32>,
}

impl Actor for DoThing {
    type Message = TaskMessage;

    fn receive_local(&mut self, msg: Self::Message) -> Handled {
        Handled::Ok
    }

    fn receive_network(&mut self, msg: kompact::prelude::NetMessage) -> Handled {
        Handled::Ok
    }
}

impl ComponentLifecycle for DoThing {
    fn on_start(&mut self) -> Handled {
        self.spawn_local(move |async_self| async move {
            State::State0 {
                a: async_self.a.clone(),
                b: async_self.b.clone(),
            }
            .await;
            Handled::DieNow
        });
        Handled::Ok
    }
}

impl DoThing {
    fn new(a: Pullable<i32>, b: Pushable<i32>) -> Self {
        Self {
            ctx: ComponentContext::uninitialised(),
            a,
            b,
        }
    }
}

fn do_thing(a: Pullable<i32>, ctx: Context) -> Pullable<i32> {
    let (b0, b1): (Pushable<i32>, Pullable<i32>) = channel(ctx);
    ctx.launch(move || DoThing::new(a, b0));
    b1
}

fn read_stream() -> Pullable<i32> {
    todo!()
}

fn main() {
    let ctx = todo!();
    let a: Pullable<i32> = read_stream();
    let b: Pullable<i32> = do_thing(a, ctx);
}
