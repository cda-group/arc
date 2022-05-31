use kompact::prelude::*;

pub struct Runtime {
    pub system: KompactSystem,
}

impl Runtime {
    pub fn new() -> Self {
        let system = KompactConfig::default().build().unwrap();
        Self { system }
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}
