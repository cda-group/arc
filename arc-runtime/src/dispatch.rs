use crate::context::Context;
use crate::data::gc::Heap;
use crate::data::Data;
use crate::manager::TaskMessage;
use async_trait::async_trait;
use std::future::Future;
use std::pin::Pin;

/// Task dispatcher
pub trait Execute: Data {
    fn execute(self, ctx: Context<Self>) -> Pin<Box<dyn Future<Output = ()> + Send>>;
}

pub fn spawn<T: Execute>(task: T, mut ctx: Context<T>) {
    let heap = Heap::new();
    let task = task.copy(heap);
    ctx.manager.tell(TaskMessage::ExecuteTask(task, heap));
}
