//! Builder interface for staging functions.

use crate::stage::partial::Partial;

/// A staged arc-script function.
#[derive(Debug, Clone)]
pub struct Fun {
    /// Name of the function.
    name: String,
    /// Input values to the function.
    args: Vec<(String, Partial)>,
}

impl Fun {
    /// Returns a new function.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            args: Vec::new(),
        }
    }
    /// Adds an argument to the function.
    pub fn arg(mut self, name: impl Into<String>, value: impl Into<Partial>) -> Self {
        self.args.push((name.into(), value.into()));
        self
    }
}
