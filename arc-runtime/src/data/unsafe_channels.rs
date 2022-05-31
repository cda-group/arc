use crate::prelude::Trace;
use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde::Serializer;
use serde_derive::Deserialize;
use serde_derive::Serialize;

use serde_state::DeserializeState;
use serde_state::SerializeState;

use std::collections::BTreeMap;

use std::collections::VecDeque;
use std::marker::PhantomData;

use crate::control::Control;
use crate::data::gc::Gc;
use crate::data::gc::Heap;
use crate::data::serde::deserialise;
use crate::data::serde::serialise;
use crate::data::serde::SerdeState;
use crate::data::Context;
use crate::data::Data;
use crate::dispatch::Execute;
use std::sync::Arc;

use crate::prelude::rewrite;
use tokio::sync::Notify;
use tokio::sync::Mutex;

pub trait Endpoint {
    type PushChan: Data;
    type PullChan: Data;
}

impl<T: Data> Endpoint for PushChan<T> {
    type PushChan = Self;
    type PullChan = PullChan<T>;
}

impl<T: Data> Endpoint for PullChan<T> {
    type PushChan = PushChan<T>;
    type PullChan = Self;
}

const CAPACITY: usize = 10;

/// An async FIFO SPSC channel.
#[derive(Debug)]
struct UnsafeChan<T: Trace> {
    queue: Mutex<VecDeque<T>>,
    heap: Heap,
    pullvar: Notify,
}

impl<T: Trace> UnsafeChan<T> {
    fn new(cap: usize) -> Self {
        Self {
            queue: Mutex::new(VecDeque::with_capacity(cap)),
            heap: Heap::new(),
            pullvar: Notify::new(),
        }
    }
}

#[derive(Copy, Clone, Debug, Trace)]
pub struct PushChan<T: Trace>(Gc<Arc<UnsafeChan<T>>>);

#[derive(Copy, Clone, Debug, Trace)]
pub struct PullChan<T: Trace>(Gc<Arc<UnsafeChan<T>>>);

pub fn channel<T: Trace>(ctx: Context<impl Execute>) -> (PushChan<T>, PullChan<T>) {
    let chan = ctx.heap.allocate(Arc::new(UnsafeChan::new(CAPACITY)));
    (PushChan(chan), PullChan(chan))
}

#[rewrite]
impl<T: Trace> PushChan<T> {
    #[allow(clippy::result_unit_err)]
    pub fn push(self, data: T, ctx: Context<impl Execute>) -> Result<(), ()> {
        let data = data.copy(self.0.heap);
        let mut queue = futures::executor::block_on(self.0.queue.lock());
        if queue.len() < queue.capacity() {
            queue.push_back(data);
            self.0.pullvar.notify_one();
            Ok(())
        } else {
            Err(())
        }
    }
}

#[rewrite]
impl<T: Trace> PullChan<T> {
    pub async fn pull(self, ctx: Context<impl Execute>) -> T {
        let mut queue = self.0.queue.lock().await;
        while queue.is_empty() {
            self.0.pullvar.notified().await;
        }
        let data = queue.pop_front().unwrap();
        data.copy(ctx.heap)
    }
}

impl<T: Data> SerializeState<SerdeState> for PushChan<T> {
    fn serialize_state<S: Serializer>(&self, _: S, _: &SerdeState) -> Result<S::Ok, S::Error> {
        // When channels are serialized and sent to another process, we must:
        // * Register the channel with the proxy
        todo!()
    }
}

impl<'de, T: Data> DeserializeState<'de, SerdeState> for PushChan<T> {
    fn deserialize_state<D: Deserializer<'de>>(_: &mut SerdeState, _: D) -> Result<Self, D::Error> {
        todo!()
    }
}

impl<T: Data> SerializeState<SerdeState> for PullChan<T> {
    fn serialize_state<S: Serializer>(&self, _: S, _: &SerdeState) -> Result<S::Ok, S::Error> {
        todo!()
    }
}

impl<'de, T: Data> DeserializeState<'de, SerdeState> for PullChan<T> {
    fn deserialize_state<D: Deserializer<'de>>(_: &mut SerdeState, _: D) -> Result<Self, D::Error> {
        todo!()
    }
}
