//! A Broadcast channel.
//! Every data item in the channel can be pulled at most once by each consumer.
//! The channel maintains an offset for each consumer, and a minimum offset.
//! This requires that the number of consumers is known in advance.
//! Can be used to implement a "data-parallel" operator.

use kompact::prelude::*;

use crate::control::Control;
use crate::data::*;
use crate::prelude::*;

use std::collections::HashMap;
use std::collections::VecDeque;

#[derive(ComponentDefinition)]
pub(crate) struct Channel<T: DynSharable> {
    ctx: ComponentContext<Self>,
    push_queue: VecDeque<Ask<T, ()>>,
    data_queue: VecDeque<T>,
    pull_queue: VecDeque<Ask<usize, T>>,
    min_offset: usize,
    offsets: HashMap<usize, usize>,
    opened: bool,
    pushers: usize,
    pullers: usize,
}

impl<T: DynSharable> Channel<T> {
    fn new() -> Self {
        Self {
            ctx: ComponentContext::uninitialised(),
            push_queue: VecDeque::with_capacity(100),
            data_queue: VecDeque::with_capacity(10),
            pull_queue: VecDeque::with_capacity(100),
            min_offset: 0,
            offsets: vec![(0, 0); 1].into_iter().collect(),
            opened: false,
            pullers: 1,
            pushers: 1,
        }
    }
}

pub fn channel<T: DynSharable>(ctx: &mut Context) -> (Pushable<T>, Pullable<T>) {
    let chan = ctx.component.system().create(Channel::new);
    ctx.component.system().start(&chan);
    (Pushable(chan.actor_ref()), Pullable(chan.actor_ref(), 1))
}

impl<T: DynSharable> ComponentLifecycle for Channel<T> {}

#[derive(Debug)]
pub(crate) enum Message<T: DynSharable> {
    PushRequest(Ask<T, ()>),
    PullRequest(Ask<usize, T>),
    Open,
    AddPusher,
    AddPuller(Ask<(), usize>),
    DelPusher,
    DelPuller(usize),
}

impl<T: DynSharable> Actor for Channel<T> {
    type Message = Message<T>;

    fn receive_local(&mut self, msg: Self::Message) -> Handled {
        match msg {
            Message::PushRequest(ask) => self.push_queue.push_back(ask),
            Message::PullRequest(ask) => self.pull_queue.push_back(ask),
            Message::Open => self.opened = true,
            Message::AddPusher => self.pushers += 1,
            Message::AddPuller(ask) => {
                ask.reply(self.pullers);
                self.pullers += 1;
            }
            Message::DelPusher => self.pushers -= 1,
            Message::DelPuller(id) => {
                self.pullers -= 1;
                self.offsets.remove(&id);
            }
        }
        if self.opened {
            while !self.push_queue.is_empty()
                && self.data_queue.len() < self.data_queue.capacity()
            {
                let (promise, data) = self.push_queue.pop_front().unwrap().take();
                promise.fulfil(()).unwrap();
                self.data_queue.push_back(data);
            }
            while !self.pull_queue.is_empty() && !self.data_queue.is_empty() {
                let (promise, id) = self.pull_queue.pop_front().unwrap().take();
                let offset = self.offsets[&id] - self.min_offset;
                let data = self.data_queue.pop_front().unwrap();
                promise.fulfil(data).unwrap();
            }
        }
        if self.pushers == 0 || self.offsets.len() == 0 {
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

pub struct Pullable<T: DynSharable>(pub(crate) ActorRef<Message<T>>, pub(crate) usize);

impl<T: DynSharable> Clone for Pushable<T> {
    fn clone(&self) -> Self {
        self.0.tell(Message::AddPusher);
        Pushable(self.0.clone())
    }
}

impl<T: DynSharable> Clone for Pullable<T> {
    fn clone(&self) -> Self {
        let id = self
            .0
            .ask_with(|promise| Message::AddPuller(Ask::new(promise, ())))
            .wait();
        Pullable(self.0.clone(), id)
    }
}

impl<T: DynSharable> Drop for Pushable<T> {
    fn drop(&mut self) {
        self.0.tell(Message::DelPusher);
    }
}

impl<T: DynSharable> Drop for Pullable<T> {
    fn drop(&mut self) {
        self.0.tell(Message::DelPuller(self.1));
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
            .ask_with(|promise| Message::PullRequest(Ask::new(promise, self.1)))
            .await
            .map(Control::Continue)
            .unwrap_or(Control::Finished)
    }
}
