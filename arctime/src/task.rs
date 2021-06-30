#![allow(clippy::type_complexity)]
#![allow(deprecated)]
#![allow(dead_code)]

use kompact::prelude::*;

use crate::port::DataReqs;
use crate::port::StreamPort;
use crate::port::ControlPort;
use crate::port::StreamEvent;
use crate::port::StreamReply;
use crate::port::ControlEvent;
use crate::port::ControlReply;
use crate::port::DateTime;
use crate::stream::*;
use crate::timer::*;

use std::sync::Arc;
use std::time::Duration;

/// A `Role` of a `Task`.
pub enum Role {
    Producer,
    ProducerConsumer,
    Consumer,
}

/// A general-purpose task component.
#[derive(ComponentDefinition)]
pub struct Task<S: DataReqs, I: DataReqs, O: DataReqs, R: DataReqs = Never> {
    pub ctx: ComponentContext<Self>,
    pub name: &'static str,
    pub promise: Option<Ask<(), R>>,
    pub data_iport: ProvidedPort<StreamPort<I>>,
    pub data_oport: RequiredPort<StreamPort<O>>,
    pub ctrl_iport: ProvidedPort<ControlPort>,
    pub ctrl_oport: RequiredPort<ControlPort>,
    pub lowest_observed_watermarks: Vec<DateTime>,
    pub time: DateTime,
    pub state: S,
    pub logic: fn(&mut Self, I),
    pub ptimer: Option<ProcessingTimer<S, I, O, R>>,
    pub etimer: EventTimer<S, I, O, R>,
    pub role: Role,
}

#[derive(Clone)]
pub struct ProcessingTimer<S: DataReqs, I: DataReqs, O: DataReqs, R: DataReqs> {
    duration: Duration,
    scheduled: Option<ScheduledTimer>,
    trigger: fn(&mut Task<S, I, O, R>),
}

impl<R: DataReqs, S: DataReqs, I: DataReqs, O: DataReqs> Actor for Task<S, I, O, R> {
    type Message = Ask<(), R>;

    fn receive_local(&mut self, msg: Self::Message) -> Handled {
        self.promise = Some(msg);
        Handled::Ok
    }

    fn receive_network(&mut self, _msg: NetMessage) -> Handled {
        todo!()
    }
}

impl<S: DataReqs, I: DataReqs, O: DataReqs, R: DataReqs> Task<S, I, O, R> {
    pub fn new(name: &'static str, state: S, logic: fn(&mut Self, I)) -> Self {
        Self {
            ctx: ComponentContext::uninitialised(),
            name,
            promise: None,
            data_iport: ProvidedPort::uninitialised(),
            data_oport: RequiredPort::uninitialised(),
            ctrl_iport: ProvidedPort::uninitialised(),
            ctrl_oport: RequiredPort::uninitialised(),
            state,
            logic,
            ptimer: None,
            role: Role::ProducerConsumer,
            lowest_observed_watermarks: vec![],
            time: DateTime::unix_epoch(),
            etimer: EventTimer::default(),
        }
    }

    pub(crate) fn set_role(self, role: Role) -> Self {
        Self { role, ..self }
    }

    pub(crate) fn new_periodic(
        name: &'static str,
        state: S,
        duration: Duration,
        trigger: fn(&mut Self),
    ) -> Self {
        Self {
            ptimer: Some(ProcessingTimer {
                duration,
                trigger,
                scheduled: None,
            }),
            ..Self::new(name, state, |_, _| {})
        }
    }

    pub(crate) fn min_watermark(&self) -> DateTime {
        *self.lowest_observed_watermarks.iter().min().unwrap()
    }

    pub fn emit(&mut self, data: O) {
        self.data_oport.trigger(StreamEvent::Data(self.time, data));
    }

    /// Die with a return value
    pub fn exit(&mut self, rval: R) {
        if let Some(promise) = self.promise.take() {
            promise.reply(rval).expect("reply");
        };
        self.data_oport.trigger(StreamEvent::End);
        self.ctx.suicide();
    }

    pub(crate) fn oneshot_trigger(&mut self, timeout: ScheduledTimer) -> Handled {
        match self.ptimer.as_mut().unwrap().scheduled.as_ref() {
            Some(scheduled_oneshot) if *scheduled_oneshot == timeout => {
                let oneshot = self.ptimer.as_ref().unwrap();
                let duration = oneshot.duration;
                (oneshot.trigger)(self);
                self.ptimer.as_mut().unwrap().scheduled =
                    Some(self.schedule_once(duration, Self::oneshot_trigger));
                Handled::Ok
            }
            Some(_) => Handled::Ok,
            None => {
                warn!(self.log(), "Got unexpected timeout: {:?}", timeout);
                Handled::Ok
            }
        }
    }
}

impl<S: DataReqs, I: DataReqs, O: DataReqs, R: DataReqs> FnOnce<(Stream<I>,)> for Task<S, I, O, R> {
    type Output = Stream<O>;

    extern "rust-call" fn call_once(self, (stream,): (Stream<I>,)) -> Self::Output {
        // Step 1. Initialise the task
        let task = stream.client.system().create(|| self);
        // Step 2. Connect the input streams to each of the task's input ports
        task.on_definition(|consumer| {
            (stream.connector)(&mut consumer.data_iport);
        });
        // Step 3. Create a stream for each of the task's output ports
        let producer = task.clone();
        let connector: Arc<ConnectFn<_>> = Arc::new(move |iport| {
            producer.on_definition(|producer| {
                iport.connect(producer.data_oport.share());
                producer.data_oport.connect(iport.share());
            });
        });
        // Step 4. Create a closure for starting up the task
        let start_fns = stream.start_fns.clone();
        let client = stream.client.clone();
        stream
            .start_fns
            .borrow_mut()
            .push(Box::new(move || client.system().start(&task)));
        let client = stream.client.clone();
        Stream::new(client, connector, start_fns)
    }
}

pub(crate) fn create_connector<S: DataReqs, I: DataReqs, O: DataReqs, R: DataReqs>(
    producer: Arc<Component<Task<S, I, O, R>>>,
) -> Arc<dyn Fn(&mut ProvidedPort<StreamPort<O>>) + 'static> {
    Arc::new(move |iport| {
        producer.on_definition(|producer| {
            iport.connect(producer.data_oport.share());
            producer.data_oport.connect(iport.share());
        })
    })
}

impl<S: DataReqs, I: DataReqs, O: DataReqs, R: DataReqs> Provide<StreamPort<I>> for Task<S, I, O, R> {
    fn handle(&mut self, event: StreamEvent<I>) -> Handled {
        match event {
            StreamEvent::Watermark(time) => {
                if time > self.time {
                    let diff = time - self.time;
                    self.advance(diff.to_std().unwrap());
                }
                Handled::Ok
            }
            StreamEvent::Data(time, data) => {
                if time >= self.time {
                    (self.logic)(self, data);
                }
                Handled::Ok
            }
            StreamEvent::End => {
                self.data_oport.trigger(StreamEvent::End);
                Handled::DieNow
            }
        }
    }
}

impl<S: DataReqs, I: DataReqs, O: DataReqs, R: DataReqs> Require<StreamPort<O>> for Task<S, I, O, R> {
    fn handle(&mut self, event: StreamReply) -> Handled {
        Handled::Ok
    }
}

impl<S: DataReqs, I: DataReqs, O: DataReqs, R: DataReqs> Provide<ControlPort> for Task<S, I, O, R> {
    fn handle(&mut self, _event: ControlEvent) -> Handled {
        todo!()
    }
}

impl<S: DataReqs, I: DataReqs, O: DataReqs, R: DataReqs> Require<ControlPort> for Task<S, I, O, R> {
    fn handle(&mut self, _: ControlReply) -> Handled {
        Handled::Ok
    }
}

impl<S: DataReqs, I: DataReqs, O: DataReqs, R: DataReqs> ComponentLifecycle for Task<S, I, O, R> {
    fn on_start(&mut self) -> Handled {
        if let Some(oneshot) = self.ptimer.as_mut() {
            let duration = oneshot.duration;
            self.ptimer.as_mut().unwrap().scheduled =
                Some(self.schedule_once(duration, Self::oneshot_trigger));
        }
        Handled::Ok
    }

    fn on_stop(&mut self) -> Handled {
        if let Some(scheduled_oneshot) = self.ptimer.as_mut().unwrap().scheduled.take() {
            self.cancel_timer(scheduled_oneshot);
        }
        Handled::Ok
    }

    fn on_kill(&mut self) -> Handled {
        Handled::Ok
    }
}
