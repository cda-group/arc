#![allow(clippy::type_complexity)]

use kompact::prelude::*;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use kompact::component::AbstractComponent;

use crate::client::*;
use crate::control::*;
use crate::data::*;
use crate::pipeline::*;
use crate::port::*;
use crate::task::*;

pub type ErasedFn = Box<dyn FnOnce()>;
pub type ConnectFn<T> = dyn Fn(&mut ProvidedPort<DataPort<T>>) + 'static;

/// A stream which can be connected to `DataPorts`.
#[derive(Clone)]
pub struct Stream<T: DataReqs> {
    pub client: Arc<Component<Client>>,
    pub connector: Arc<dyn Fn(&mut ProvidedPort<DataPort<T>>) + 'static>,
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
