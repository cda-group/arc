#![allow(clippy::type_complexity)]

use kompact::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use crate::client::*;
use crate::port::*;

pub type ErasedFn = Box<dyn FnOnce()>;
pub type ConnectFn<T> = dyn Fn(&mut ProvidedPort<StreamPort<T>>) + 'static;

/// A stream which can be connected to `DataPorts`.
#[derive(Clone)]
pub struct Stream<T: DataReqs> {
    pub client: Arc<Component<Client>>,
    pub connector: Arc<dyn Fn(&mut ProvidedPort<StreamPort<T>>) + 'static>,
    pub start_fns: Rc<RefCell<Vec<ErasedFn>>>,
}

impl<I: DataReqs> Stream<I> {
    pub fn new(
        client: Arc<Component<Client>>,
        connect: Arc<ConnectFn<I>>,
        starters: Rc<RefCell<Vec<ErasedFn>>>,
    ) -> Self {
        Self {
            client,
            connector: connect,
            start_fns: starters,
        }
    }
}
