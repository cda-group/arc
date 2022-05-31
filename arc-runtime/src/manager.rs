use crate::context::Context;
use crate::data::channels::PushChan;
use crate::data::gc::Heap;
use crate::data::serde::{deserialise, serialise};
use crate::dispatch::Execute;
use kompact::config_keys::system::THREADS;
use kompact::executors::crossbeam_workstealing_pool::ThreadPool;
use kompact::executors::parker;
use kompact::prelude::*;
use kompact::serde_serialisers::Serde;
use serde::Deserialize;
use serde::Serialize;
use serde_derive::Deserialize;
use serde_derive::Serialize;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::Mutex;
use tokio::runtime::Runtime;
use tokio::spawn;

#[derive(ComponentDefinition)]
pub struct Manager<E: Execute> {
    ctx: ComponentContext<Self>,
    runtime: Runtime,
    peers: Vec<ActorPath>,
    task_count: usize,
    /// Outgoing channels. When we receive a `TaskMessage::PushData(id, data)` we push it to the channel with the same id.
    outgoing: HashMap<ChanId, ActorPath>,
    /// Ingoing channels. When we receive a `ManagerMessage::PushData(id, data)` we push it to the channel with the same id.
    ingoing: HashMap<ChanId, PushChan<Vec<u8>>>,
    main: fn((), Context<E>),
}

#[derive(Debug)]
pub enum TaskMessage<E: Execute> {
    ExecuteTask(E, Heap),
    CompleteTask,
    PushData(ChanId, Vec<u8>),
    RelocateTask(Vec<u8>),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChanId(usize);

#[derive(Debug, Serialize, Deserialize)]
pub enum ManagerMessage {
    ExecuteTask(Vec<u8>),
    PushData(ChanId, Vec<u8>),
}

impl SerialisationId for ManagerMessage {
    const SER_ID: u64 = 0;
}

enum Location {
    Local,
    Remote,
}

impl<E: Execute> Manager<E> {
    fn location(&self) -> Location {
        if self.peers.is_empty() {
            Location::Local
        } else {
            // TODO: Check pressure of this machine
            Location::Remote
        }
    }
}

impl<E: Execute> Actor for Manager<E> {
    type Message = TaskMessage<E>;
    // Receive message from a local task
    fn receive_local(&mut self, msg: Self::Message) -> Handled {
        match msg {
            // Execute a task from the manager.
            TaskMessage::ExecuteTask(execute, heap) => {
                self.task_count += 1;
                let ctx = self.create_ctx_with_heap(heap);
                match self.location() {
                    Location::Local => drop(self.runtime.spawn(execute.execute(ctx))),
                    Location::Remote => drop(self.runtime.spawn(async move {
                        let _guard = ctx.guard();
                        ctx.manager
                            .tell(TaskMessage::RelocateTask(serialise(execute, ctx.serde)));
                    })),
                }
            }
            // Send task to another manager
            TaskMessage::RelocateTask(data) => {
                self.task_count -= 1;
                self.peers
                    .first()
                    .unwrap()
                    .tell((ManagerMessage::ExecuteTask(data), Serde), self);
            }
            // Register that a task was completed by a task.
            TaskMessage::CompleteTask => {
                self.task_count -= 1;
                if self.task_count == 0 {
                    self.ctx().system().shutdown_async();
                }
            }
            TaskMessage::PushData(chan, data) => {
                // self.channels[chan.0].push(data);
            }
        }
        Handled::Ok
    }
    // Receive request from a remote manager
    fn receive_network(&mut self, msg: NetMessage) -> Handled {
        match msg.data.try_deserialise::<ManagerMessage, Serde>().unwrap() {
            ManagerMessage::ExecuteTask(data) => {
                let ctx = self.create_ctx();
                self.runtime
                    .spawn(deserialise::<E>(data, ctx.serde).execute(ctx));
            }
            ManagerMessage::PushData(chan, data) => {}
        }
        Handled::Ok
    }
}

impl<E: Execute> Manager<E> {
    fn new(runtime: Runtime, main: fn((), Context<E>)) -> Self {
        Self {
            ctx: ComponentContext::uninitialised(),
            peers: Vec::new(),
            runtime,
            task_count: 0,
            main,
            outgoing: HashMap::new(),
            ingoing: HashMap::new(),
        }
    }
    fn create_ctx(&self) -> Context<E> {
        Context::new(self.ctx.actor_ref())
    }
    fn create_ctx_with_heap(&self, heap: Heap) -> Context<E> {
        Context::new_with_heap(heap, self.ctx.actor_ref())
    }
}

impl<E: Execute> ComponentLifecycle for Manager<E> {
    fn on_start(&mut self) -> Handled {
        let ctx = self.create_ctx();
        let guard = ctx.guard();
        (self.main)((), ctx);
        if ctx.task_count == 0 {
            self.ctx.system().shutdown_async();
        }
        Handled::Ok
    }
}

impl<E: Execute> Manager<E> {
    pub fn start(main: fn((), Context<E>)) {
        // Split cores between tokio and kompact
        let mut cores = core_affinity::get_core_ids().unwrap();
        cores.reverse();
        let kompact_core = if cores.len() > 1 {
            cores.pop().unwrap()
        } else {
            cores[0]
        };
        let n = cores.len();
        let tokio_cores = Arc::new(Mutex::new(cores));

        // Setup Tokio
        let tokio = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(n)
            .max_blocking_threads(1)
            .on_thread_start(move || {
                if let Some(core) = tokio_cores.lock().unwrap().pop() {
                    core_affinity::set_for_current(core);
                }
            })
            .build()
            .unwrap();

        // Setup Kompact
        let mut config = KompactConfig::default();
        config.set_config_value(&THREADS, 1);
        config.executor(move |_| ThreadPool::with_affinity(&[kompact_core], 0, parker::small()));
        let system = config.build().unwrap();

        // Setup Manager
        let manager = system.create(move || Manager::new(tokio, main));
        system.register(&manager).wait();
        system.start(&manager);
        system.await_termination();
    }
}
