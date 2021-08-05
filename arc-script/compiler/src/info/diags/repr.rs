use crate::ast::lower::source::lexer::sem_tokens::Token;
use crate::ast::Name;

use crate::hir::Path;
use crate::hir::HIR;
use crate::info::diags::to_codespan::Context;
use crate::info::diags::to_codespan::ToCodespan;
use crate::info::files::Loc;

use crate::info::types::Type;
use crate::info::Info;

use std::io::Write;

use codespan_reporting::term;

use codespan_reporting::term::termcolor::Color;

use codespan_reporting::term::termcolor::ColorSpec;

use arc_script_compiler_shared::From;
use arc_script_compiler_shared::Shrinkwrap;
use codespan_reporting::term::termcolor::WriteColor;
use codespan_reporting::term::Config;

type CodespanResult = std::result::Result<(), codespan_reporting::files::Error>;

/// Interner for storing diagnostics.
#[derive(Shrinkwrap, Debug, Default)]
#[shrinkwrap(mutable)]
pub(crate) struct DiagInterner {
    /// Store containing all diagnostics interned so far. Multiple stores can
    /// coexist in different interners, e.g., when parsing, and may be merged
    /// into one interner.
    pub(crate) store: Vec<Diagnostic>,
}

impl DiagInterner {
    /// Add a new diagnostic.
    pub(crate) fn intern(&mut self, d: impl Into<Diagnostic>) {
        self.store.push(d.into());
    }

    /// Merges two sets of diagnostics.
    pub(crate) fn merge(&mut self, mut other: Self) {
        self.store.append(&mut other.store)
    }

    /// Returns `true` if the interner is empty, otherwise `false`.
    pub(crate) fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Drains all diagnostics of the interner and places them into a new interner.
    pub(crate) fn take(&mut self) -> Self {
        std::mem::take(self)
    }

    /// Emits all diagnostics in the interner. The HIR does not need to be passed if the
    /// diagnostics were generated only while parsing.
    pub(crate) fn emit<'i, W>(
        &self,
        info: &Info,
        hir: impl Into<Option<&'i HIR>>,
        f: &mut W,
    ) -> CodespanResult
    where
        W: Write + WriteColor,
    {
        // Emit header message
        f.set_color(ColorSpec::new().set_fg(Some(Color::Red)));
        writeln!(f, "[-- Found {} errors --],", self.len())?;
        tracing::trace!("{:?}", self);
        f.reset();

        let files = &info.files.store;
        let config = &Config::default();
        let ctx = &Context::new(info, hir.into());
        self.iter()
            .filter_map(|diag| diag.to_codespan(ctx))
            .try_for_each(|diag| term::emit(f, config, files, &diag))
    }
}

/// A diagnostic result.
pub(crate) type Result<T> = std::result::Result<T, Diagnostic>;

/// A compile-time or runtime diagnostic.
#[derive(Debug, From)]
pub enum Diagnostic {
    /// A compile-time note
    Note(Note),
    /// A compile-time warning
    Warning(Warning),
    /// A compile-time error
    Error(Error),
}

/// Compile-time info reported by the compiler.
#[derive(Debug)]
pub enum Note {}

/// Compile-time warnings reported by the compiler.
#[derive(Debug)]
pub enum Warning {}

/// Compile-time errors reported by the compiler.
#[derive(Debug)]
pub enum Error {
    /// Error when the importer fails to find a source file.
    FileNotFound,

    /// Error when the importing is given a file with an invalid extension.
    BadExtension,

    /// Error when the lexer encounters too many levels of indentation.
    TooMuchIndent {
        /// Location of the error.
        loc: Loc,
    },

    /// Error when the lexer encounters indentation indivisible by TABSIZE.
    BadIndent {
        /// Location of the error.
        loc: Loc,
    },

    /// Error produced by [`lexical_core`].
    LexicalCore {
        /// Error code.
        err: lexical_core::Error,
        /// Location of the error.
        loc: Loc,
    },

    /// Error produced by [`time`].
    Time {
        /// Error code.
        err: time::ParseError,
        /// Location of the error.
        loc: Loc,
    },

    /// Error when the parser comes across an extra token.
    ExtraToken {
        /// Extra token found while parsing.
        found: Token,
        /// Location of the error.
        loc: Loc,
    },

    /// Error when the parser comes across an invalid token.
    InvalidToken {
        /// Location of the error.
        loc: Loc,
    },

    /// Error when the parser comes across an unrecognized end-of-file.
    UnrecognizedEOF {
        /// Location of the error.
        loc: Loc,
        /// List of tokens expected by LALRPOP.
        expected: Vec<String>,
    },

    /// Error when the parser comes across an unrecognized token.
    /// In other words, a token emitted by the lexer which the parser
    /// did not expect.
    /// NB: This is technically an Internal-Compiler-Error.
    UnrecognizedToken {
        /// Unrecognized token found while parsing.
        found: Token,
        /// Location of the token.
        loc: Loc,
        /// List of tokens expected by LALRPOP.
        expected: Vec<String>,
    },

    /// Error when two types fail to unify.
    TypeMismatch {
        /// First type.
        lhs: Type,
        /// Second type.
        rhs: Type,
        /// Location of the error.
        loc: Loc,
    },

    /// Error when type information is needed at a code location, but is not supplied.
    TypeMustBeKnownAtThisPoint {
        /// Location of the error.
        loc: Loc,
    },

    /// Error when a path does not reference anything.
    PathNotFound {
        /// Path which failed to resolve.
        path: Path,
        /// Location of the error.
        loc: Loc,
    },

    /// Error when a match is non-exhaustive.
    NonExhaustiveMatch {
        /// Location of the error.
        loc: Loc,
    },

    /// Error when two items in the same namespace have the same name.
    NameClash {
        /// Name of the two items.
        name: Name,
    },

    /// Error when a struct contains two fields with the same name.
    FieldClash {
        /// Name of the two fields.
        name: Name,
    },

    /// Error when an enum contains two variants with the same name.
    VariantClash {
        /// Name of the two variants.
        name: Name,
    },

    /// Error when an enum variant is constructed with too many arguments.
    VariantWrongArity {
        /// Path of the enum variant.
        path: Path,
    },

    /// Error when a tuple is indexed with an out-of-bounds index.
    OutOfBoundsProject {
        /// Location of the error.
        loc: Loc,
    },

    /// Error when a struct-field is accessed which does not exist.
    FieldNotFound {
        /// Location of the error.
        loc: Loc,
    },

    /// Error when placing a type in value position.
    TypeInValuePosition {
        /// Location of the error.
        loc: Loc,
    },

    /// Error when placing a value in type position.
    ValueInTypePosition {
        /// Location of the error.
        loc: Loc,
    },

    /// Error when enwrapping a non-variant path.
    PathIsNotVariant {
        /// Location of the error.
        loc: Loc,
    },

    /// Error when placing a refutable pattern where an irrefutable one is expected.
    RefutablePattern {
        /// Location of the error.
        loc: Loc,
    },

    /// Error when moving a used value, and the moved value does not implement Copy.
    UseOfMovedValue {
        /// Location of the parent value.
        loc0: Loc,
        /// Location of the second value.
        loc1: Loc,
        /// Type which does not implement Copy.
        t: Type,
    },

    /// Error when using the same value twice, and the used value does not implement Copy.
    DoubleUse {
        /// Location of the place expression declaration.
        loc0: Loc,
        /// Location of the first use.
        loc1: Loc,
        /// Location of the second use.
        loc2: Loc,
        /// Type which does not implement Copy.
        t: Type,
    },

    /// Error when an extern function contains a parameter which is a pattern.
    PatternInExternFun {
        /// Location of the extern function.
        loc: Loc,
    },

    /// Error when an expression expects a selector (e.g., e1 not in e2[123]), but the selector is
    /// not specified.
    ExpectedSelector {
        /// Location of the expression
        loc: Loc,
    },

    /// Error when multiple selectors are specified. For now this is a hard-error, but will relaxed
    /// in the future.
    MultipleSelectors {
        /// Location of the expression
        loc: Loc,
    },

    /// Error when using a non-selectable type (map, set) in a selector.
    ExpectedSelectableType {
        /// Location of the expression
        loc: Loc,
    },
}

/// Runtime errors reported by the compiler.
#[derive(Debug)]
pub enum Panic {
    /// Stack-trace to code location which caused a panic.
    Unwind {
        /// Location of the runtime error.
        loc: Loc,
        /// Path to each function in the call-stack.
        trace: Vec<Path>,
    },
}
