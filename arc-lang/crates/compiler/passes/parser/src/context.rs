use diagnostics::Diagnostics;
use sources::Sources;

#[derive(Debug, Default)]
pub struct Context {
    pub diagnostics: Diagnostics,
    pub sources: Sources,
}

impl Context {
    pub fn new() -> Self {
        Self {
            diagnostics: Diagnostics::default(),
            sources: Sources::new(),
        }
    }

    pub fn id(&self) -> usize {
        self.sources.len()
    }
}
