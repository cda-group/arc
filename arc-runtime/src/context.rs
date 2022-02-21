use comet::immix::Immix;
use comet::immix::ImmixOptions;
use comet::mutator::MutatorRef;
use derive_more::Constructor as New;
use kompact::prelude::*;

use crate::prelude::Send;
use crate::prelude::Sync;
use crate::prelude::Unpin;
use std::sync::Arc;

/// The context of a single task.
#[derive(Copy, Clone, Send, Sync, Unpin)]
pub struct Context(*mut Core);

/// The data stored by the context.
#[derive(New)]
struct Core {
    pub component: Arc<dyn CoreContainer>,
    pub mutator: MutatorRef<Immix>,
}

impl Context {
    #[allow(clippy::mut_from_ref)]
    fn as_mut(&self) -> &mut Core {
        // SAFETY: This is safe because the context is only ever accessed from a single task.
        unsafe { &mut *self.0 }
    }
}

impl Context {
    pub fn new(component: Arc<dyn CoreContainer>, mutator: MutatorRef<Immix>) -> Self {
        Self(Box::leak(Box::new(Core::new(component, mutator))) as *mut Core)
    }
    pub fn destroy(self) {
        // SAFETY: This is safe because the context is managed entirely by the code generator. This
        // function is only ever called once.
        unsafe {
            Box::from_raw(self.0);
        }
    }
    #[allow(clippy::mut_from_ref)]
    pub fn mutator(&self) -> &mut MutatorRef<Immix> {
        &mut self.as_mut().mutator
    }
    #[allow(clippy::mut_from_ref)]
    pub fn component(&self) -> &mut Arc<dyn CoreContainer> {
        &mut self.as_mut().component
    }
    pub fn launch<C, F>(&self, f: F)
    where
        F: FnOnce() -> C,
        C: ComponentDefinition + 'static,
    {
        let system = self.as_mut().component.system();
        let c = system.create(f);
        system.start(&c);
    }
}
