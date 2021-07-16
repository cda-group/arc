use kompact::prelude::*;

use crate::pipeline::*;
use crate::port::*;
use crate::stream::*;
use crate::task::*;

impl<I: DataReqs> Stream<I> {
    /// Write a stream to a sink.
    /// Also returns a new pipeline (which can for example be finalized)
    pub fn sink<S: DataReqs>(self, task: Task<S, I, Never, Never>) -> Pipeline<impl SystemHandle> {
        let stream = self.apply(task);
        Pipeline {
            system: stream.client.on_definition(|c| c.ctx().system()),
            client: stream.client,
            startup: stream.start_fns,
        }
    }

    /// Write a stream to a sink.
    /// Also returns a new pipeline (which can for example be finalized)
    pub fn log_sink(self) -> Pipeline<impl SystemHandle> {
        let task = Task::new(
            "Logger",
            (),
            |task: &mut Task<(), I, Never, Never>, event| {
                info!(task.ctx.log(), "{:?}", event);
            },
        );
        let stream = self.apply(task);
        Pipeline {
            system: stream.client.on_definition(|c| c.ctx().system()),
            client: stream.client,
            startup: stream.start_fns,
        }
    }
}
