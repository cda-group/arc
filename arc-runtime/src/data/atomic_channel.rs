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
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

const CAPACITY: usize = 10;

use std::cell::UnsafeCell;
use tokio::sync::Notify;

/// An async FIFO SPSC channel.
#[derive(Debug)]
struct AtomicChan<T> {
    queue: UnsafeCell<VecDeque<T>>,
    notifier: Notify,
    count: AtomicUsize,
}

impl<T> AtomicChan<T> {
    fn new(cap: usize) -> Self {
        Self {
            queue: UnsafeCell::new(VecDeque::with_capacity(cap)),
            notifier: Notify::new(),
            count: AtomicUsize::new(0),
        }
    }
    #[allow(clippy::mut_from_ref)]
    fn queue(&self) -> &mut VecDeque<T> {
        unsafe { &mut *self.queue.get() }
    }
}

#[derive(Debug)]
pub struct PushChan<T>(Arc<AtomicChan<T>>);

#[derive(Debug)]
pub struct PullChan<T>(Arc<AtomicChan<T>>);

pub fn channel<T>() -> (PushChan<T>, PullChan<T>) {
    let chan = Arc::new(AtomicChan::new(CAPACITY));
    (PushChan(chan.clone()), PullChan(chan))
}

impl<T> PushChan<T> {
    pub fn push(self, data: T) {
        if self.0.count.load(Ordering::SeqCst) == self.0.queue().capacity() {
            panic!("channel is full");
        }
        self.0.queue().push_back(data);
        self.0.count.fetch_add(1, Ordering::SeqCst);
        self.0.notifier.notify_one();
    }
}

impl<T> PullChan<T> {
    pub async fn pull(self) -> T {
        while self.0.count.load(Ordering::SeqCst) == 0 {
            self.0.notifier.notified().await;
        }
        let data = self.0.queue().pop_front().unwrap();
        self.0.count.fetch_sub(1, Ordering::SeqCst);
        data
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
