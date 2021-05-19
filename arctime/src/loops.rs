#![allow(clippy::type_complexity)]

use kompact::component::AbstractComponent;
use kompact::prelude::*;

use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use crate::client::*;
use crate::control::*;
use crate::data::*;
use crate::pipeline::*;
use crate::port::*;
use crate::stream::*;
use crate::task::*;

impl<I: DataReqs> Stream<I> {
    /// Apply a transformation to a stream to produce a new stream.
    /// ```ignore
    /// stream
    ///     .iterate(|s: Stream<i32>| {
    ///        let s1 = Filter(|x| x < 100) (s.clone());
    ///        let s2 = Map(x| x + 1) (s1);
    ///        let s3 = Filter(|x| x >= 100) (s);
    ///        (s1, s3)
    ///    })
    /// ```
    pub(crate) fn iterate<O: DataReqs>(
        self,
        f: fn(Stream<I>) -> (Stream<I>, Stream<O>),
    ) -> Stream<O> {
        let task = Task::new(
            "LoopHead",
            (),
            |task: &mut Task<(), I, I, Never>, event: I| {
                task.emit(event);
            },
        );
        let start_fns = self.start_fns.clone();
        let client = self.client.clone();
        let task = self.client.system().create(|| task);
        task.on_definition(|consumer| (self.connector)(&mut consumer.data_iport));
        let producer = task.clone();
        let connector: Arc<ConnectFn<_>> = Arc::new(move |iport| {
            producer.on_definition(|producer| {
                iport.connect(producer.data_oport.share());
                producer.data_oport.connect(iport.share());
            });
        });
        let task_feedback = task.clone();
        start_fns
            .borrow_mut()
            .push(Box::new(move || client.system().start(&task)));
        let client = self.client.clone();
        let stream = Stream::new(client, connector, start_fns);
        let (feedback, output) = f(stream);
        task_feedback.on_definition(|consumer| (feedback.connector)(&mut consumer.data_iport));
        output
    }

    fn structured_loop<R: DataReqs, O: DataReqs, const A: usize>(
        self,
        body: fn(Stream<I>, Stream<R>) -> (Stream<R>, Stream<O>),
        cond: fn(R, Scope<A>) -> bool,
    ) -> Stream<O> {
        todo!()
        // Iteration Head => Termination condition
        // Iteration Tail
        // Entry
        // Exit
        // Progress = [pn, pn-1, pn-2, .. p0] = [Current|Context]
        // P ≥ P′ iff (Pctx = P′ctx) ∧ (PT ≥ P′T)
    }
}

struct Scope<const A: usize>([i32; A]);
