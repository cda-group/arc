#![allow(unused_mut)]
#![allow(unreachable_code)]
#![allow(unused_variables)]
#![feature(arbitrary_self_types)]
use arc_runtime::data::channels::local::multicast::*;
use arc_runtime::prelude::*;

#[derive(ComponentDefinition)]
struct DoThing {
    ctx: ComponentContext<Self>,
    a: Pullable<i32>,
    b: Pullable<i32>,
    c: Pushable<i32>,
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

impl DoThing {
    fn new(a: Pullable<i32>, b: Pullable<i32>, c: Pushable<i32>) -> Self {
        Self {
            ctx: ComponentContext::uninitialised(),
            a,
            b,
            c,
        }
    }

    async fn run(mut self: ComponentDefinitionAccess<Self>, ctx: Context) -> Control<()> {
        loop {
            let x = self.a.pull(ctx).await?;
            let y = self.b.pull(ctx).await?;
            self.c.push(x + y, ctx).await?;
        }
        Control::Finished
    }
}

fn do_thing(a: Pullable<i32>, b: Pullable<i32>, ctx: Context) -> Pullable<i32> {
    let (c0, c1): (Pushable<i32>, Pullable<i32>) = channel(ctx);
    let task = ctx.component().system().create(move || DoThing::new(a, b, c0));
    ctx.component().system().start(&task);
    c1
}

impl ComponentLifecycle for DoThing {
    fn on_start(&mut self) -> Handled {
        self.spawn_local(move |mut async_self| async move {
            let ctx = todo!();
            async_self.run(ctx).await;
            Handled::DieNow
        });
        Handled::Ok
    }
}

fn read_stream() -> Pullable<i32> {
    todo!()
}

fn main() {
    let ctx = todo!();
    let a: Pullable<i32> = read_stream();
    let b: Pullable<i32> = read_stream();
    let c = do_thing(a, b, ctx);
}
