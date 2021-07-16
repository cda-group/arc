#![allow(clippy::type_complexity)]
use kompact::prelude::*;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use crate::client::*;
use crate::port::*;
use crate::executor::*;
use crate::task::*;

/// A pipeline of tasks.
pub struct Pipeline<S: SystemHandle> {
    pub(crate) system: S,
    pub(crate) client: Arc<Component<Client>>,
    pub(crate) startup: Rc<RefCell<Vec<Box<dyn FnOnce() + 'static>>>>,
}

impl<S: SystemHandle> Pipeline<S> {
    /// Finalizes the construction of a pipeline by starting up all the tasks.
    pub fn build(self) {
        for starter in self.startup.borrow_mut().drain(..).rev() {
            starter();
        }
    }
}

impl Executor {
    pub fn pipeline(&self) -> Pipeline<impl SystemHandle> {
        let starters = Rc::new(RefCell::new(Vec::new()));
        let client = self.system.create(Client::new);
        let system = client.on_definition(|c| c.ctx().system());
        Pipeline {
            system,
            client,
            startup: starters,
        }
    }
}

impl<S: DataReqs, I: DataReqs, O: DataReqs, R: DataReqs> Task<S, I, O, R> {
    pub fn pipeline(&self) -> Pipeline<impl SystemHandle> {
        let system = self.ctx.system();
        let client = system.create(Client::new);
        let starters = Rc::new(RefCell::new(Vec::new()));
        Pipeline {
            system,
            client,
            startup: starters,
        }
    }
}
