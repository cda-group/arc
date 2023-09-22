use std::ops::Range;

use text_size::TextRange;
use text_size::TextSize;

/// A structure which keeps the start and end position of an AST node plus its source.
pub type Spanned<T> = (T, Info);

/// An index of a character in a source.
pub type ByteIndex = TextSize;

/// A span between two characters in a source.
pub type Span = TextRange;

/// An identifier of a source. Can be used to access the source code of the source.
pub type SourceId = usize;

/// Stores a code location.
#[derive(Debug, Clone, Copy)]
pub enum Info {
    Source { id: SourceId, span: Span },
    Builtin,
}

impl Info {
    pub fn new(id: SourceId, start: ByteIndex, end: ByteIndex) -> Self {
        Self::Source {
            id,
            span: Span::new(start, end),
        }
    }

    /// Constructs a source from an id and a byte index range.
    pub fn from_range(id: SourceId, range: Range<ByteIndex>) -> Self {
        Self::Source {
            id,
            span: Span::new(range.start, range.end),
        }
    }

    /// Joins two locations into a potentially larger location.
    pub fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Source { id, span }, Self::Source { span: span2, .. }) => Self::Source {
                id,
                span: span.cover(span2),
            },
            (Self::Builtin, x) | (x, Self::Builtin) => x,
        }
    }

    pub fn span(self) -> Range<usize> {
        match self {
            Self::Source { span, .. } => span.start().into()..span.end().into(),
            Self::Builtin => 0..0,
        }
    }

    pub fn id(self) -> Option<SourceId> {
        match self {
            Self::Source { id, .. } => Some(id),
            Self::Builtin => None,
        }
    }
}
