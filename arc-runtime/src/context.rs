use crate::data::gc::Heap;
use crate::manager::TaskMessage;
use derive_more::Constructor as New;
use kompact::prelude::*;
use rskafka::client::Client as Kafka;
use serde_json::Value;

use crate::data::serde::SerdeState;
use crate::dispatch::Execute;
use crate::prelude::Send;
use crate::prelude::Sync;
use crate::prelude::Unpin;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

/// The context of a single task.
#[derive(Send, Sync, Unpin)]
pub struct Context<E: Execute>(*mut Core<E>);

impl<E: Execute> Copy for Context<E> {}
impl<E: Execute> Clone for Context<E> {
    fn clone(&self) -> Self {
        *self
    }
}

/// The data stored by the context.
pub struct Core<E: Execute> {
    pub heap: Heap,
    pub manager: ActorRef<TaskMessage<E>>,
    pub serde: SerdeState,
    pub partition: i32,
    pub task_count: usize,
    pub phantom: std::marker::PhantomData<E>,
}

impl<E: Execute> std::ops::Deref for Context<E> {
    type Target = Core<E>;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.0 }
    }
}

impl<E: Execute> std::ops::DerefMut for Context<E> {
    fn deref_mut(&mut self) -> &mut Core<E> {
        unsafe { &mut *self.0 }
    }
}

impl<E: Execute> Core<E> {
    pub fn new(heap: Heap, manager: ActorRef<TaskMessage<E>>) -> Self {
        Self {
            heap,
            manager,
            serde: SerdeState::new(heap),
            partition: 0,
            task_count: 0,
            phantom: std::marker::PhantomData,
        }
    }

    pub fn execute(&mut self, task: E, heap: Heap) {
        self.task_count += 1;
        self.manager.tell(TaskMessage::ExecuteTask(task, heap));
    }
}

impl<E: Execute> Context<E> {
    pub fn new(manager: ActorRef<TaskMessage<E>>) -> Self {
        Self::new_with_heap(Heap::new(), manager)
    }
    pub fn new_with_heap(heap: Heap, manager: ActorRef<TaskMessage<E>>) -> Self {
        Self(Box::into_raw(Box::new(Core::new(heap, manager))))
    }
    pub fn guard(self) -> Guard<E> {
        Guard { ctx: self }
    }
}

pub struct Guard<E: Execute> {
    ctx: Context<E>,
}

impl<E: Execute> Drop for Guard<E> {
    fn drop(&mut self) {
        self.ctx.manager.tell(TaskMessage::CompleteTask);
        unsafe {
            Box::from_raw(self.ctx.0);
        }
    }
}
