use arc_script_core_shared::New;

use codespan_reporting::files::SimpleFiles;
use text_size::TextRange;
use text_size::TextSize;

use std::ops::Range;

/// A struct for storing source files.
#[derive(Debug)]
pub struct FileInterner {
    /// Store containing source names and source files. Can be used
    pub store: SimpleFiles<String, String>,
}

impl Default for FileInterner {
    fn default() -> Self {
        Self {
            store: SimpleFiles::new(),
        }
    }
}

/// An index of a character in a source file.
pub type ByteIndex = TextSize;

/// A span between two characters in a source file.
pub type Span = TextRange;

/// An identifier of a source file. Can be used to access the source code of the source file.
pub type FileId = usize;

/// Stores a code location.
#[derive(Debug, Clone, Copy)]
pub enum Loc {
    /// A real location which maps to a span in a source file.
    Real(FileId, Span),
    /// A fake location, synthesized during compilation.
    Fake,
}

impl Loc {
    /// Constructs a source file from a file and a byte index range.
    pub(crate) fn from_range(file: FileId, range: Range<ByteIndex>) -> Self {
        Self::Real(file, Span::new(range.start, range.end))
    }

    /// Joins two locations into a potentially larger location.
    pub(crate) fn join(self, other: Self) -> Self {
        match (self, other) {
            (Loc::Real(file, span0), Loc::Real(_, span1)) => Self::Real(file, span0.cover(span1)),
            (loc @ Loc::Real(..), Loc::Fake) | (Loc::Fake, loc @ Loc::Real(..)) => loc,
            (Loc::Fake, Loc::Fake) => Loc::Fake,
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
