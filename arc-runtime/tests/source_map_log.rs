#![allow(unused)]
#![feature(arbitrary_self_types)]
#![allow(unused_mut)]

macro_rules! compile_test {
    {$($mod:tt)::+} => {
        use arc_runtime::prelude::*;

        // NOTE: The `where` clause is not necessary when we have monomorphised the code.
        #[derive(ComponentDefinition)]
        struct Source<T: Sharable> where T::T: Sendable<T = T> {
            ctx: ComponentContext<Self>,
            vec: Vec<T>,
            pushable: $($mod)::+::Pushable<T>,
        }

        #[derive(ComponentDefinition)]
        struct Map<A: Sharable, B: Sharable> where A::T: Sendable<T = A>, B::T: Sendable<T = B> {
            ctx: ComponentContext<Self>,
            pullable: $($mod)::+::Pullable<A>,
            fun: fn(A) -> B,
            pushable: $($mod)::+::Pushable<B>,
        }

        #[derive(ComponentDefinition)]
        struct Log<T: Sharable> where T::T: Sendable<T = T> {
            ctx: ComponentContext<Self>,
            pullable: $($mod)::+::Pullable<T>,
        }

        impl<T: Sharable> Source<T> where T::T: Sendable<T = T> {
            fn new(vec: Vec<T>, pushable: $($mod)::+::Pushable<T>) -> Self {
                Self {
                    ctx: ComponentContext::uninitialised(),
                    vec,
                    pushable,
                }
            }

            async fn run(mut self: ComponentDefinitionAccess<Self>, ctx: Context) -> Control<()> {
                let i = self.vec.clone();
                for x in 0..i.clone().len(ctx) {
                    let j = i.clone();
                    let v = j.at(x, ctx);
                    self.pushable.push(v.clone(), ctx).await?;
                }
                Control::Finished
            }
        }

        impl<A: Sharable, B: Sharable> Map<A, B> where A::T: Sendable<T=A>, B::T: Sendable<T=B> {
            fn new(pullable: $($mod)::+::Pullable<A>, f: fn(A) -> B, pushable: $($mod)::+::Pushable<B>) -> Self {
                Self {
                    ctx: ComponentContext::uninitialised(),
                    pullable,
                    fun: f,
                    pushable,
                }
            }

            async fn run(mut self: ComponentDefinitionAccess<Self>, ctx: Context) -> Control<()> {
                let f = self.fun;
                loop {
                    let data = self.pullable.pull(ctx).await?;
                    self.pushable.push(f(data), ctx).await?;
                }
            }
        }

        impl<T: Sharable> Log<T> where T::T: Sendable<T=T> {
            fn new(pullable: $($mod)::+::Pullable<T>) -> Self {
                Self {
                    ctx: ComponentContext::uninitialised(),
                    pullable,
                }
            }

            async fn run(mut self: ComponentDefinitionAccess<Self>, ctx: Context) -> Control<()> {
                loop {
                    let data = self.pullable.pull(ctx).await?;
                    info!(self.log(), "Logging {:?}", data);
                }
            }
        }

        impl<T: Sharable> ComponentLifecycle for Source<T> where T::T: Sendable<T=T> {
            fn on_start(&mut self) -> Handled {
                self.spawn_local(move |async_self| async move {
                    let component = async_self.ctx().component();
                    let mutator = instantiate_immix(ImmixOptions::default());
                    let ctx = Context::new(component, mutator);
                    async_self.run(ctx).await;
                    Handled::DieNow
                });
                Handled::Ok
            }
        }

        impl<A: Sharable, B: Sharable> ComponentLifecycle for Map<A, B> where A::T: Sendable<T=A>, B::T: Sendable<T=B> {
            fn on_start(&mut self) -> Handled {
                self.spawn_local(move |async_self| async move {
                    let component = async_self.ctx().component();
                    let mutator = instantiate_immix(ImmixOptions::default());
                    let ctx = Context::new(component, mutator);
                    async_self.run(ctx).await;
                    Handled::DieNow
                });
                Handled::Ok
            }
        }

        impl<T: Sharable> ComponentLifecycle for Log<T> where T::T: Sendable<T=T> {
            fn on_start(&mut self) -> Handled {
                self.spawn_local(move |async_self| async move {
                    let component = async_self.ctx().component();
                    let mutator = instantiate_immix(ImmixOptions::default());
                    let ctx = Context::new(component, mutator);
                    async_self.run(ctx).await;
                    Handled::DieNow
                });
                Handled::Ok
            }
        }

        impl<T: Sharable> Actor for Source<T> where T::T: Sendable<T=T> {
            type Message = TaskMessage;

            fn receive_local(&mut self, _msg: Self::Message) -> Handled {
                Handled::Ok
            }

            fn receive_network(&mut self, _msg: NetMessage) -> Handled {
                unreachable!()
            }
        }

        impl<A: Sharable, B: Sharable> Actor for Map<A, B> where A::T: Sendable<T=A>, B::T: Sendable<T=B> {
            type Message = TaskMessage;

            fn receive_local(&mut self, _msg: Self::Message) -> Handled {
                Handled::Ok
            }

            fn receive_network(&mut self, _msg: NetMessage) -> Handled {
                unreachable!()
            }
        }

        impl<T: Sharable> Actor for Log<T> where T::T: Sendable<T=T> {
            type Message = TaskMessage;

            fn receive_local(&mut self, _msg: Self::Message) -> Handled {
                Handled::Ok
            }

            fn receive_network(&mut self, _msg: NetMessage) -> Handled {
                unreachable!()
            }
        }

        fn source<T: Sharable>(vec: Vec<T>, ctx: Context) -> $($mod)::+::Pullable<T> where T::T: Sendable<T=T> {
            let (o0, o1) = $($mod)::+::channel(ctx);
            ctx.launch(move || Source::new(vec, o0));
            o1
        }

        fn map<A: Sharable, B: Sharable>(a: $($mod)::+::Pullable<A>, f: fn(A) -> B, ctx: Context) -> $($mod)::+::Pullable<B> where A::T: Sendable<T=A>, B::T: Sendable<T=B> {
            let (b0, b1) = $($mod)::+::channel(ctx);
            ctx.launch(move || Map::new(a, f, b0));
            b1
        }

        fn log<T: Sharable>(a: $($mod)::+::Pullable<T>, ctx: Context) where T::T: Sendable<T=T> {
            ctx.launch(move || Log::new(a));
        }

        fn plus_one(x: i32) -> i32 {
            x + 1
        }

        #[derive(ComponentDefinition, Actor)]
        struct Main {
            ctx: ComponentContext<Self>,
        }

        impl Main {
            fn new() -> Self {
                Self {
                    ctx: ComponentContext::uninitialised()
                }
            }
        }

        fn run_main(ctx: Context) {
            let v = vector![1i32, 2, 3];
            let v = source(v, ctx);
            let v = map(v, plus_one, ctx);
            let _ = log(v, ctx);
        }

        impl ComponentLifecycle for Main {
            fn on_start(&mut self) -> Handled {
                let component = self.ctx().component();
                let mutator = instantiate_immix(ImmixOptions::default());
                let ctx = Context::new(component, mutator);
                run_main(ctx);
                self.ctx().system().shutdown_async();
                Handled::DieNow
            }
        }

        fn main() {
            let system = KompactConfig::default().build().unwrap();
            let main = system.create(move || Main::new());
            system.start(&main);
            system.await_termination();
        }
    }
}

// mod source_map_log_remote_concurrent {
//     compile_test!(arc_runtime::data::channels::remote::concurrent);
// }

// mod source_map_log_remote_broadcast {
//     compile_test!(arc_runtime::data::channels::remote::broadcast);
// }

// mod source_map_log_local_concurrent {
//     compile_test!(arc_runtime::data::channels::local::task_parallel);
// }

mod source_map_log_local_broadcast {
    compile_test!(arc_runtime::data::channels::local::multicast);
}
