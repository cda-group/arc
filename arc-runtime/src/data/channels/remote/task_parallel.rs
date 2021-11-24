use kompact::component::AbstractComponent;
use kompact::prelude::*;
use time::*;

use crate::control::Control;
use crate::data::*;
use crate::prelude::*;

use std::collections::HashMap;
use std::collections::VecDeque;
use std::ops::ControlFlow;
use std::ops::FromResidual;
use std::ops::Try;
use std::sync::Arc;
use std::time::Duration;

#[derive(ComponentDefinition)]
pub(crate) struct Channel<T: DynSharable> {
    ctx: ComponentContext<Self>,
    push_queue: VecDeque<Ask<T, ()>>,
    data_queue: VecDeque<T>,
    pull_queue: VecDeque<Ask<(), T>>,
    pushers: usize,
    pullers: usize,
}

impl<T: DynSharable> Channel<T> {
    pub fn new() -> Self {
        Channel {
            ctx: ComponentContext::uninitialised(),
            push_queue: VecDeque::with_capacity(100),
            data_queue: VecDeque::with_capacity(10),
            pull_queue: VecDeque::with_capacity(100),
            pullers: 1,
            pushers: 1,
        }
    }
}

pub fn channel<T: DynSharable>(ctx: &mut Context) -> (Pushable<T>, Pullable<T>) {
    let chan = ctx.component.system().create(Channel::new);
    ctx.component.system().start(&chan);
    (Pushable(chan.actor_ref()), Pullable(chan.actor_ref()))
}

impl<T: DynSharable> ComponentLifecycle for Channel<T> {}

#[derive(Debug)]
pub(crate) enum Message<T: DynSharable> {
    PushRequest(Ask<T, ()>),
    PullRequest(Ask<(), T>),
    AddPusher,
    AddPuller,
    DelPusher,
    DelPuller,
}

impl<T: DynSharable> Actor for Channel<T> {
    type Message = Message<T>;

    fn receive_local(&mut self, msg: Self::Message) -> Handled {
        match msg {
            Message::PushRequest(ask) => self.push_queue.push_back(ask),
            Message::PullRequest(ask) => self.pull_queue.push_back(ask),
            Message::AddPusher => self.pushers += 1,
            Message::AddPuller => self.pullers += 1,
            Message::DelPusher => self.pushers -= 1,
            Message::DelPuller => self.pullers -= 1,
        }
        while !self.push_queue.is_empty() && self.data_queue.len() < self.data_queue.capacity() {
            let (promise, data) = self.push_queue.pop_front().unwrap().take();
            promise.fulfil(()).unwrap();
            self.data_queue.push_back(data);
        }
        while !self.pull_queue.is_empty() && !self.data_queue.is_empty() {
            let (promise, id) = self.pull_queue.pop_front().unwrap().take();
            let data = self.data_queue.pop_front().unwrap();
            promise.fulfil(data).unwrap();
        }
        if self.pushers == 0 && self.data_queue.is_empty() || self.pullers == 0 {
            Handled::DieNow
        } else {
            Handled::Ok
        }
    }

    fn receive_network(&mut self, msg: NetMessage) -> Handled {
        todo!()
    }
}

pub struct Pushable<T: DynSharable>(pub(crate) ActorRef<Message<T>>);

pub struct Pullable<T: DynSharable>(pub(crate) ActorRef<Message<T>>);

impl<T: DynSharable> Clone for Pushable<T> {
    fn clone(&self) -> Self {
        self.0.tell(Message::AddPusher);
        Pushable(self.0.clone())
    }
}

impl<T: DynSharable> Clone for Pullable<T> {
    fn clone(&self) -> Self {
        Pullable(self.0.clone())
    }
}

impl<T: DynSharable> Drop for Pushable<T> {
    fn drop(&mut self) {
        self.0.tell(Message::DelPusher);
    }
}

impl<T: DynSharable> Drop for Pullable<T> {
    fn drop(&mut self) {
        self.0.tell(Message::DelPuller);
    }
}

impl<T: DynSharable> Pushable<T> {
    pub async fn push(&self, data: T) -> Control<()> {
        self.0
            .ask_with(|promise| Message::PushRequest(Ask::new(promise, data)))
            .await
            .map(Control::Continue)
            .unwrap_or(Control::Finished)
    }
}

impl<T: DynSharable> Pullable<T> {
    pub async fn pull(&self) -> Control<T> {
        self.0
            .ask_with(|promise| Message::PullRequest(Ask::new(promise, ())))
            .await
            .map(Control::Continue)
            .unwrap_or(Control::Finished)
    }
}
