#![feature(arbitrary_self_types)]
#![allow(unused_mut)]
#![allow(unreachable_code)]

use arc_runtime::channels::local::concurrent::channel;
use arc_runtime::channels::local::concurrent::Pullable;
use arc_runtime::channels::local::concurrent::Pushable;
use arc_runtime::prelude::*;

lazy_static::lazy_static! {
    static ref EXECUTOR: Executor = Executor::new();
}

#[derive(Actor, ComponentDefinition)]
struct Source<I: IntoIterator<Item = T> + Data, T: Data>
where
    <I as IntoIterator>::IntoIter: Data,
{
    ctx: ComponentContext<Self>,
    iter: I,
    pushable: Pushable<T>,
}
impl<I: IntoIterator<Item = T> + Data, T: Data> Source<I, T>
where
    <I as IntoIterator>::IntoIter: Data,
{
    fn new(iter: I, pushable: Pushable<T>) -> Self {
        Self {
            ctx: ComponentContext::uninitialised(),
            iter,
            pushable,
        }
    }

    async fn run(mut self: ComponentDefinitionAccess<Self>) -> Control<()> {
        let i = self.iter.clone();
        for x in i {
            self.pushable.push(x).await?;
        }
        Control::Finished
    }
}

#[derive(Actor, ComponentDefinition)]
struct Log<T: Data> {
    ctx: ComponentContext<Self>,
    pullable: Pullable<T>,
}

impl<T: Data> Log<T> {
    fn new(pullable: Pullable<T>) -> Self {
        Self {
            ctx: ComponentContext::uninitialised(),
            pullable,
        }
    }

    async fn run(mut self: ComponentDefinitionAccess<Self>) -> Control<()> {
        loop {
            let data = self.pullable.pull().await?;
//             info!(self.log(), "{:?}", data);
        }
    }
}

impl<I: IntoIterator<Item = T> + Data, T: Data> ComponentLifecycle for Source<I, T>
where
    <I as IntoIterator>::IntoIter: Data,
{
    fn on_start(&mut self) -> Handled {
        self.spawn_local(move |async_self| async move {
            async_self.run().await;
            Handled::DieNow
        });
        Handled::Ok
    }
}

impl<T: Data> ComponentLifecycle for Log<T> {
    fn on_start(&mut self) -> Handled {
        self.spawn_local(move |async_self| async move {
            async_self.run().await;
            Handled::DieNow
        });
        Handled::Ok
    }
}
fn source<I: Data, T: Data>(i: I) -> Pullable<T>
where
    I: IntoIterator<Item = T>,
    <I as IntoIterator>::IntoIter: Data,
{
    let (a0, a1) = channel(&EXECUTOR);
    EXECUTOR.create_task(move || Source::new(i, a0));
    a1
}

fn log<T: Data>(a: Pullable<T>) {
    EXECUTOR.create_task(move || Log::new(a));
}

fn read_stream() -> Pullable<i32> {
    todo!()
}

fn master() {
    log(source(0..100));
}

fn worker() {
    todo!()
}

fn main() {
    EXECUTOR.init({
        let mut cfg = KompactConfig::default();
        cfg.load_config_file("./application.conf");
        cfg.system_components(DeadletterBox::new, NetworkConfig::default().build());
        cfg.build().expect("KompactSystem")
    });

    match std::env::args().nth(1).as_ref().map(|x| x.as_str()) {
        Some("--master") => master(),
        Some("--worker") => worker(),
        _ => panic!("Expected --master or --worker"),
    }
    EXECUTOR.await_termination();
}
