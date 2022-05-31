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
use crate::data::gc::Trace;
use crate::data::serde::deserialise;
use crate::data::serde::serialise;
use crate::data::serde::SerdeState;
use crate::data::Context;
use crate::data::Data;
use crate::dispatch::Execute;
use std::sync::Arc;

use crate::prelude::rewrite;
use async_std::sync::Condvar;
use async_std::sync::Mutex;

const CAPACITY: usize = 10;

#[derive(Debug)]
struct DynChan {
    queue: Mutex<VecDeque<Vec<u8>>>,
    heap: Heap,
    pullvar: Condvar,
    pushvar: Condvar,
}

#[derive(Copy, Clone, Debug)]
pub struct DynPushChan(Arc<DynChan>);

#[derive(Copy, Clone, Debug)]
pub struct DynPullChan(Arc<DynChan>);

impl DynChan {
    fn new(cap: usize) -> Self {
        Self {
            queue: Mutex::new(VecDeque::with_capacity(cap)),
            heap: Heap::new(),
            pullvar: Condvar::new(),
            pushvar: Condvar::new(),
        }
    }
}

pub fn channel(ctx: Context<impl Execute>) -> (DynPushChan, DynPullChan) {
    let chan = ctx.heap.allocate(Arc::new(DynChan::new(CAPACITY)));
    (DynPushChan(chan), DynPullChan(chan))
}

impl<T: Data> DynPushChan<T> {
    pub async fn push(self, data: Vec<u8>, ctx: Context<impl Execute>) {
        let data = deserialise::<T>(data, ctx.serde);
        let mut queue = self.0.queue.lock().await;
        while queue.len() == queue.capacity() {
            queue = self.0.pushvar.wait(queue).await;
        }
        queue.push_back(data);
        self.0.pullvar.notify_one();
    }
}

impl<T: Data> DynPullChan<T> {
    pub async fn pull(self, ctx: Context<impl Execute>) -> T {
        let mut queue = self.0.queue.lock().await;
        while queue.is_empty() {
            queue = self.0.pullvar.wait(queue).await;
        }
        let data = queue.pop_front().unwrap().copy(ctx.heap);
        self.0.pushvar.notify_one();
        data
    }
}

impl<T: Data> SerializeState<SerdeState> for DynPushChan<T> {
    fn serialize_state<S: Serializer>(&self, _: S, _: &SerdeState) -> Result<S::Ok, S::Error> {
        // When channels are serialized and sent to another process, we must:
        // * Register the channel with the proxy
        todo!()
    }
}

impl<'de, T: Data> DeserializeState<'de, SerdeState> for DynPushChan<T> {
    fn deserialize_state<D: Deserializer<'de>>(_: &mut SerdeState, _: D) -> Result<Self, D::Error> {
        todo!()
    }
}

impl<T: Data> SerializeState<SerdeState> for DynPullChan<T> {
    fn serialize_state<S: Serializer>(&self, _: S, _: &SerdeState) -> Result<S::Ok, S::Error> {
        todo!()
    }
}

impl<'de, T: Data> DeserializeState<'de, SerdeState> for DynPullChan<T> {
    fn deserialize_state<D: Deserializer<'de>>(_: &mut SerdeState, _: D) -> Result<Self, D::Error> {
        todo!()
    }
}
