use crate::compiler::ast::from::lexer::Token;
use crate::compiler::ast::Name;
use crate::compiler::hir;
use crate::compiler::hir::Path;
use crate::compiler::hir::HIR;
use crate::compiler::info::diags::to_codespan::{Context, ToCodespan};
use crate::compiler::info::files::Loc;
use crate::compiler::info::paths::PathId;
use crate::compiler::info::types::TypeId;
use crate::compiler::info::Info;

use std::io;
use std::io::Write;
use std::str;

use codespan_reporting::diagnostic::{self, Label};
use codespan_reporting::term::termcolor::{
    Buffer, Color, ColorChoice, ColorSpec, StandardStream, WriteColor,
};
use codespan_reporting::term::{self, Config};

type CodespanResult = std::result::Result<(), codespan_reporting::files::Error>;

#[derive(Shrinkwrap, Debug, Default)]
#[shrinkwrap(mutable)]
pub struct DiagInterner {
    pub store: Vec<Diagnostic>,
}

impl DiagInterner {
    /// Adds a diagnostic.
    pub(crate) fn intern(&mut self, d: impl Into<Diagnostic>) {
        self.store.push(d.into());
    }

    /// Merges two sets of diagnostics.
    pub(crate) fn merge(&mut self, mut other: Self) {
        self.store.append(&mut other.store)
    }

    /// Appends two sets of diagnostics.
    pub(crate) fn extend(&mut self, other: impl IntoIterator<Item = Diagnostic>) {
        self.store.extend(other)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
}

/// A diagnostic result.
pub(crate) type Result<T> = std::result::Result<T, Diagnostic>;

/// A compile-time diagnostic. May either be a warning or error.
#[derive(Debug)]
pub enum Diagnostic {
    Note(Note),
    Warning(Warning),
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

    /// Error when the lexer encounters too many levels of indentation.
    TooMuchIndent { loc: Option<Loc> },

    /// Error when the lexer encounters indentation indivisible by TABSIZE.
    BadIndent { loc: Option<Loc> },

    LexicalCore {
        err: lexical_core::Error,
        loc: Option<Loc>,
    },

    /// Error when the lexer fails to parse a literal.
    BadLiteral { msg: String, loc: Option<Loc> },

    /// Error when the parser comes across an extra token.
    ExtraToken { found: Token, loc: Option<Loc> },

    /// Error when the parser comes across an invalid token.
    InvalidToken { loc: Option<Loc> },

    /// Error when the parser comes across an unrecognized end-of-file.
    UnrecognizedEOF {
        loc: Option<Loc>,
        expected: Vec<String>,
    },

    /// Error when the parser comes across an unexpected token.
    UnrecognizedToken {
        found: Token,
        loc: Option<Loc>,
        expected: Vec<String>,
    },

    /// Error when two types fail to unify.
    TypeMismatch {
        lhs: TypeId,
        rhs: TypeId,
        loc: Option<Loc>,
    },

    /// Error when a path does not reference anything.
    PathNotFound { path: Path, loc: Option<Loc> },

    /// Error when a path does not reference anything.
    DisallowedDimExpr { loc: Option<Loc> },

    /// Error when z3 cannot infer anything about a shape.
    ShapeUnknown,

    /// Error when z3 runs into a contradiction when inferring a shape.
    ShapeUnsat,

    /// Error when a match is non-exhaustive.
    NonExhaustiveMatch { loc: Option<Loc> },

    /// Error when two items in the same namespace have the same name.
    NameClash { name: Name },

    /// Error when a struct contains two fields with the same name.
    FieldClash { name: Name },

    /// Error when an enum contains two fields with the same name.
    VariantClash { name: Name },

    /// Error when a tuple is indexed with an out-of-bounds index.
    OutOfBoundsProject { loc: Option<Loc> },

    /// Error when a struct-field is accessed which does not exist.
    FieldNotFound { loc: Option<Loc> },

    /// Error when the main function has the wrong signature.
    MainWrongSign,

    /// Error when there is no main-function.
    MainNotFound,

    /// Error when a cycle is detected in the DFG.
    CycleDetected,

    /// Error when placing a type in value position.
    TypeInValuePosition { loc: Option<Loc> },

    /// Error when enwrapping a non-variant path.
    PathIsNotVariant { loc: Option<Loc> },

    /// Error when placing a certain pattern inside a let-expression.
    RefutablePattern { loc: Option<Loc> },

    /// Error when moving a used value.
    UseOfMovedValue {
        loc0: Option<Loc>,
        loc1: Option<Loc>,
    },

    /// Error when using the same value twice.
    DoubleUse {
        loc0: Option<Loc>,
        loc1: Option<Loc>,
        loc2: Option<Loc>,
    },

    /// Runtime error.
    Panic { loc: Option<Loc>, trace: Vec<Path> },
}

impl DiagInterner {
    pub(crate) fn take(&mut self) -> DiagInterner {
        std::mem::take(self)
    }
    pub(crate) fn emit<W>(&self, info: &Info, hir: Option<&HIR>, f: &mut W) -> CodespanResult
    where
        W: Write + WriteColor,
    {
        f.set_color(ColorSpec::new().set_fg(Some(Color::Red)));
        writeln!(f, "[-- Found {} errors --],", self.len())?;
        f.reset();

        let files = &info.files.store;
        let config = &Config::default();
        let ctx = &Context { info, hir };
        self.iter()
            .filter_map(|diag| diag.to_codespan(ctx))
            .try_for_each(|diag| term::emit(f, config, files, &diag))
    }
}
