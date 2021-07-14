#![allow(clippy::type_complexity)]
#![allow(deprecated)]
#![allow(dead_code)]

use arrayvec::ArrayVec;
use kompact::component::AbstractComponent;
use kompact::prelude::*;
use time::*;

use crate::control::*;
use crate::pipeline::*;
use crate::port::*;
use crate::stream::*;
use crate::timer::*;

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

#[derive(ComponentDefinition)]
pub struct ChannelComponent<T: DataReqs> {
    ctx: ComponentContext<Self>,
    push_queue: VecDeque<T>,
    pull_queue: VecDeque<Ask<(), T>>,
    time: DateTime,
}

impl<T: DataReqs> Default for ChannelComponent<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: DataReqs> ChannelComponent<T> {
    pub fn new() -> Self {
        Self {
            ctx: ComponentContext::uninitialised(),
            push_queue: VecDeque::default(),
            pull_queue: VecDeque::default(),
            time: DateTime::unix_epoch(),
        }
    }
}

impl<T: DataReqs> ComponentLifecycle for ChannelComponent<T> {
    fn on_start(&mut self) -> Handled {
        Handled::Ok
    }

    fn on_stop(&mut self) -> Handled {
        Handled::Ok
    }

    fn on_kill(&mut self) -> Handled {
        Handled::Ok
    }
}

#[derive(Debug)]
pub enum ChannelMessage<T: DataReqs> {
    Push(T),
    Pull(Ask<(), T>),
}

impl<T: DataReqs> Actor for ChannelComponent<T> {
    type Message = ChannelMessage<T>;

    fn receive_local(&mut self, msg: Self::Message) -> Handled {
        match msg {
            ChannelMessage::Push(data) => self.push_queue.push_back(data),
            ChannelMessage::Pull(ask) => self.pull_queue.push_back(ask),
        }
        while !self.push_queue.is_empty() && !self.pull_queue.is_empty() {
            let data = self.push_queue.pop_back().unwrap();
            let ask = self.pull_queue.pop_back().unwrap();
            ask.reply(data).expect("Ask reply failed");
        }
        Handled::Ok
    }

    fn receive_network(&mut self, msg: NetMessage) -> Handled {
        todo!()
    }
}

struct Channel<T: DataReqs>(ActorRef<ChannelMessage<T>>);

impl<T: DataReqs> Channel<T> {
    fn push(&self, data: T) {
        self.0.tell(ChannelMessage::Push(data));
    }

    fn pull(&self) -> KFuture<T> {
        self.0
            .ask_with(|promise| ChannelMessage::Pull(Ask::new(promise, ())))
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn basic() {
        let system = KompactConfig::default().build().expect("system");
        let chan0 = system.create(ChannelComponent::<i32>::new);
        let chan1 = system.create(ChannelComponent::<i32>::new);
        system.start(&chan0);
        system.start(&chan1);
        let chan0 = Channel(chan0.actor_ref());
        let chan1 = Channel(chan1.actor_ref());
        test(chan0, chan1);
        system.await_termination();
    }

    async fn test(chan0: Channel<i32>, chan1: Channel<i32>) {
        for x in 0..100 {
            chan0.push(3);
        }
        for _ in 0..100 {
            let data = chan0.pull().await.unwrap();
            chan1.push(data);
        }
    }
}
