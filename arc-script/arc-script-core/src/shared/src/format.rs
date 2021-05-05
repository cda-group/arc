use crate::New;
use shrinkwraprs::Shrinkwrap;

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

/// A generic formatter which can be used during pretty printing to store indentation.
#[derive(New, Copy, Clone, Shrinkwrap)]
pub struct Format<T: Copy + Clone> {
    #[shrinkwrap(main_field)]
    pub ctx: T,
    indentation: u32,
}

impl<T: Copy + Clone> AsRef<Format<T>> for Format<T> {
    fn as_ref(&self) -> &Self {
        self
    }
}

/// The space of a tab character.
pub(crate) const TABSPACE: &str = "    ";

impl<T: Copy + Clone> Display for Format<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        (0..self.indentation).try_for_each(|_| write!(f, "{}", TABSPACE))
    }
}

impl<T: Copy + Clone> Format<T> {
    /// Creates a new typeset with the specified ctx.
    pub fn with_ctx(ctx: T) -> Self {
        Self {
            ctx,
            indentation: 0,
        }
    }
    /// Returns a new typeset which has +1 indentation.
    pub fn indent(&self) -> Self {
        Self::new(self.ctx, self.indentation + 1)
    }

    /// Returns a new typeset which has -1 indentation.
    #[allow(unused)]
    pub fn dedent(&self) -> Self {
        Self::new(self.ctx, self.indentation - 1)
    }
}
