use crate::compiler::shared::New;

use std::ops::Range;

#[derive(Debug)]
pub struct FileInterner {
    /// TODO: Should we store String or can we store &str here?
    /// Maybe insignificant when mem::swap is involved.
    pub store: codespan_reporting::files::SimpleFiles<String, String>,
}

impl Default for FileInterner {
    fn default() -> Self {
        Self {
            store: codespan_reporting::files::SimpleFiles::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, New)]
pub struct Span(ByteIndex, ByteIndex);

impl From<Span> for Range<ByteIndex> {
    fn from(span: Span) -> Self {
        span.0..span.1
    }
}

pub type FileId = usize;

/// A position in the source code.
pub type ByteIndex = usize;

/// A structure which stores a code location.
#[derive(Debug, Clone, Copy)]
pub struct Loc {
    pub file: FileId,
    pub span: Span,
}

impl Loc {
    pub(crate) fn new(file: FileId, range: Range<ByteIndex>) -> Self {
        Self {
            file,
            span: Span::new(range.start, range.end),
        }
    }

    pub(crate) fn join(self, other: Loc) -> Self {
        Self {
            file: self.file,
            span: Span(self.span.0, other.span.1),
        }
    }
}

impl FileInterner {
    /// Interns a file's name and source to a `FileId`.
    pub(crate) fn intern(&mut self, name: String, source: String) -> FileId {
        self.store.add(name, source)
    }

    /// Resolves a `FileId` to its source.
    pub(crate) fn resolve(&mut self, id: FileId) -> &str {
        self.store.get(id).unwrap().source()
    }
}
