use kompact::prelude::*;

/// The core data structure for creating and executing pipelines.
pub struct Executor {
    pub(crate) system: KompactSystem,
}

impl Executor {
    pub fn new() -> Self {
        Executor {
            system: KompactConfig::default().build().expect("system"),
        }
    }

    pub fn execute(self) {
        self.system.await_termination()
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}
