use kompact::prelude::*;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::Arc;

use kompact::component::AbstractComponent;

use crate::client::*;
use crate::control::*;
use crate::data::*;
use crate::pipeline::*;
use crate::port::*;
use crate::stream::*;
use crate::task::*;

impl<I: DataReqs> Stream<I> {
    /// Write a stream to a sink.
    /// Also returns a new pipeline (which can for example be finalized)
    pub fn sink<S: DataReqs>(
        self,
        task: Task<S, I, Never, Never>,
    ) -> Pipeline<impl SystemHandle> {
        let stream = self.apply(task);
        Pipeline {
            system: stream.client.on_definition(|c| c.ctx().system()),
            client: stream.client,
            startup: stream.start_fns,
        }
    }
}
