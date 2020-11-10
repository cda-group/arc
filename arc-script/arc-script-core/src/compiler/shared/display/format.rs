use crate::compiler::ast;
use crate::compiler::info;
use crate::compiler::shared::New;

use std::fmt::{self, Display, Formatter};

/// A generic context which can be used during pretty printing to store indentation.
#[derive(New, Copy, Clone)]
pub(crate) struct Context<T: Copy + Clone> {
    pub(crate) state: T,
    indentation: u32,
}

impl<T: Copy + Clone> AsRef<Context<T>> for Context<T> {
    fn as_ref(&self) -> &Self {
        self
    }
}

pub(crate) const TABSPACE: &str = "    ";

impl<T: Copy + Clone> Display for Context<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        (0..self.indentation).try_for_each(|_| write!(f, "{}", TABSPACE))
    }
}

impl<T: Copy + Clone> Context<T> {
    /// Creates a new context with the specified state.
    pub(crate) fn with_state(state: T) -> Self {
        Self {
            state,
            indentation: 0,
        }
    }
    /// Returns a new context which has +1 indentation.
    pub(crate) fn indent(&self) -> Self {
        Self::new(self.state, self.indentation + 1)
    }

    /// Returns a new context which has -1 indentation.
    #[allow(unused)]
    pub(crate) fn dedent(&self) -> Self {
        Self::new(self.state, self.indentation - 1)
    }
}
