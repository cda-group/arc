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

pub use text_size::TextRange as Span;
pub use text_size::TextSize as ByteIndex;
pub type FileId = usize;

/// A structure which stores a code location.
#[derive(Debug, Clone, Copy, New)]
pub struct Loc {
    pub file: FileId,
    pub span: Span,
}

impl Loc {
    pub(crate) fn from_range(file: FileId, range: Range<ByteIndex>) -> Self {
        Self::new(file, Span::new(range.start, range.end))
    }

    pub(crate) fn join(self, other: Loc) -> Self {
        Self::new(self.file, self.span.cover(other.span))
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
