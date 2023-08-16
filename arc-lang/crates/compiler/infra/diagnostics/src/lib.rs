#![allow(unused)]

mod builder;

use std::borrow::Cow;
use std::fmt::Display;
use std::path::PathBuf;

use ariadne::Cache;
use ariadne::Label;
use ariadne::Report;
use ariadne::ReportKind;
use ariadne::Source;
use config::Config;
use info::Info;
use sources::Sources;

#[derive(Debug, Clone, Default)]
pub struct Diagnostics {
    pub backtrace: bool,
    pub failfast: bool,
    pub(crate) errors: Vec<Error>,
    pub(crate) warnings: Vec<Warning>,
    pub(crate) hints: Vec<Hint>,
}

impl Diagnostics {
    pub fn new(backtrace: bool, failfast: bool) -> Self {
        Self {
            backtrace,
            failfast,
            errors: Vec::new(),
            warnings: Vec::new(),
            hints: Vec::new(),
        }
    }

    #[track_caller]
    pub fn push_error(&mut self, error: Error) {
        if self.backtrace {
            println!("{}", std::panic::Location::caller());
        }
        if self.failfast {
            println!("{:?}", error);
            std::process::exit(1);
        }
        self.errors.push(error);
    }

    pub fn push_warning(&mut self, warning: Warning) {
        self.warnings.push(warning);
    }

    pub fn push_hint(&mut self, hint: Hint) {
        self.hints.push(hint);
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    pub fn has_hints(&self) -> bool {
        !self.hints.is_empty()
    }

    pub fn is_empty(&self) -> bool {
        self.errors.is_empty() && self.warnings.is_empty() && self.hints.is_empty()
    }

    pub fn report_if_some(&mut self, sources: &mut Sources, opt: &Config) {
        if self.has_errors() {
            self.emit_errors(sources, opt);
        }
        if self.has_warnings() {
            self.emit_warnings(sources, opt);
        }
        if self.has_hints() {
            self.emit_hints(sources, opt);
        }
    }

    pub fn emit(&mut self, sources: &mut Sources, opt: &Config) {
        self.emit_errors(sources, opt);
        if opt.show.warnings {
            self.emit_warnings(sources, opt);
        }
        if opt.show.hints {
            self.emit_hints(sources, opt);
        }
    }

    pub fn emit_errors(&mut self, sources: &mut Sources, opt: &Config) {
        for error in self.errors.drain(..) {
            if error.emit(sources).is_none() {
                eprintln!("Internal Compiler Error: {:?}", error);
            }
        }
    }

    pub fn emit_warnings(&mut self, sources: &mut Sources, opt: &Config) {
        for warning in self.warnings.drain(..) {
            if warning.emit(sources).is_none() {
                eprintln!("Internal Compiler Warning: {:?}", warning);
            }
        }
    }

    pub fn emit_hints(&mut self, sources: &mut Sources, opt: &Config) {
        for hint in self.hints.drain(..) {
            if hint.emit(sources).is_none() {
                eprintln!("Internal Compiler Hint: {:?}", hint);
            }
        }
    }

    pub fn append(&mut self, other: &mut Self) {
        self.errors.append(&mut other.errors);
        self.warnings.append(&mut other.warnings);
        self.hints.append(&mut other.hints);
    }

    pub fn take(&mut self) -> Self {
        Self {
            backtrace: self.backtrace,
            failfast: self.failfast,
            errors: std::mem::take(&mut self.errors),
            warnings: std::mem::take(&mut self.warnings),
            hints: std::mem::take(&mut self.hints),
        }
    }
}

/// Compile-time info reported by the compiler.
#[derive(Debug, Clone)]
pub enum Hint {}

/// Compile-time warnings reported by the compiler.
#[derive(Debug, Clone)]
pub enum Warning {
    UnusedVariable {
        /// Name of the unused variable.
        name: String,
        /// Location of the unused variable.
        info: Info,
    },
    ShadowedVariable {
        info0: Info,
        info1: Info,
    },
}

/// Compile-time errors reported by the compiler.
#[derive(Debug, Clone)]
pub enum Error {
    /// Error when the importer fails to find a source file.
    FileNotFound {
        path: PathBuf,
    },

    /// Error when the lexer comes across an invalid token.
    LexerInvalidToken {
        /// Location of the error.
        info: Info,
        err: LexerError,
    },

    /// Error when the parser comes across an extra token.
    ParserExtraToken {
        /// Extra token found while parsing.
        found: String,
        /// Location of the error.
        info: Info,
    },

    /// Error when the parser comes across an unrecognized end-of-file.
    ParserUnrecognizedEof {
        /// Location of the error.
        info: Info,
        /// List of tokens expected by LALRPOP.
        expected: Vec<String>,
    },

    /// Error when the parser comes across an unrecognized token.
    ParserUnrecognizedToken {
        /// Unrecognized token found while parsing.
        found: String,
        /// Location of the token.
        info: Info,
        /// List of tokens expected by LALRPOP.
        expected: Vec<String>,
    },

    DuplicateMetaKey {
        /// Location of the error.
        info: Info,
        /// Name of the duplicate key.
        key: String,
    },

    GenericWithArgs {
        /// Location of the error.
        info: Info,
        /// Name of the generic type.
        name: String,
    },

    UnresolvedTypeName {
        /// Location of the error.
        info: Info,
        /// Name of the unresolved type.
        name: String,
    },

    ExpectedVariant {
        info: Info,
    },

    /// Error when two types fail to unify.
    TypeMismatch {
        /// First type.
        lhs: String,
        /// Second type.
        rhs: String,
        /// Location of the error.
        info: Info,
    },

    /// Error when a match is non-exhaustive.
    NonExhaustiveMatch {
        /// Location of the error.
        info: Info,
    },

    /// Error when two items in the same namespace have the same name.
    NameClash {
        info0: Info,
        info1: Info,
    },

    /// Error when a struct contains two fields with the same name.
    DuplicateRow {
        info: Info,
    },

    /// Error when placing a type in value position.
    TypeInValuePosition {
        info: Info,
    },

    /// Error when placing a value in type position.
    ValueInTypePosition {
        info: Info,
    },

    /// Error when a function is expected, but a type or class is found.
    ExpectedFunction {
        info: Info,
    },

    /// Error when a path cannot be resolved.
    UnresolvedName {
        info: Info,
    },

    ExpectedPlaceExpr {
        info: Info,
    },
    UnexpectedTypeArgs {
        info: Info,
    },

    ExpectedVar {
        info: Info,
    },
    DuplicateVariant {
        info: Info,
    },
    BreakOutsideInfiniteLoop {
        info: Info,
    },
    BreakOutsideLoop {
        info: Info,
    },
    ContinueOutsideLoop {
        info: Info,
    },
    GenericError {
        info: Info,
        text: Cow<'static, str>,
    },
    InfiniteType {
        info: Info,
        t: String,
    },
    WrongNumberOfTypeArgs {
        name: String,
        expected: usize,
        found: usize,
        info0: Info,
        info1: Info,
    },
    UncompileableCode {
        info: Info,
        msg: &'static str,
    },
    ExpectedVarOrVal {
        info: Info,
    },
    NameNotFound {
        info: Info,
        name: String,
    },
    ExpectedIrrefutablePattern {
        info: Info,
    },
    RowNotFound {
        info: Info,
        x: String,
    },
    InterpreterError {
        info: Info,
        s: &'static str,
    },
    WrongNumberOfArguments {
        info: Info,
        expected: usize,
        found: usize,
    },
    SynError {
        info: Info,
        e: syn::Error,
    },
}

impl Hint {
    fn emit(&self, _sources: &mut Sources) -> Option<()> {
        Some(())
    }
}

impl Warning {
    fn emit(&self, sources: &mut Sources) -> Option<()> {
        match self {
            Warning::UnusedVariable { name, info } => {
                Report::build(ReportKind::Warning, info.id()?, 0)
                    .with_message("Unused variable")
                    .with_label(
                        Label::new((info.id()?, info.span()))
                            .with_message(&format!("Variable `{name}` is unused")),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Warning::ShadowedVariable { info0, info1 } => {
                Report::build(ReportKind::Warning, info0.id()?, 0)
                    .with_message("Shadowed variable")
                    .with_label(
                        Label::new((info0.id()?, info0.span()))
                            .with_message(&format!("Variable is shadowed")),
                    )
                    .with_label(
                        Label::new((info1.id()?, info1.span()))
                            .with_message(&format!("Shadowed by this variable")),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
        };
        Some(())
    }
}

impl Error {
    pub fn emit(&self, sources: &mut Sources) -> Option<()> {
        match self {
            Error::FileNotFound { path } => {
                eprintln!("File not found: {}", path.display())
            }
            Error::ParserExtraToken { found, info } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Unused variable")
                    .with_label(
                        Label::new((info.id()?, info.span()))
                            .with_message(&format!("Variable `{}` is unused", "")),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::LexerInvalidToken { info, err } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Invalid token")
                    .with_label(Label::new((info.id()?, info.span())).with_message(err))
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::ParserUnrecognizedEof { info, expected } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Unexpected end of file")
                    .with_label(
                        Label::new((info.id()?, info.span()))
                            .with_message("Unexpected end of file"),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::ParserUnrecognizedToken {
                found,
                info,
                expected,
            } => {
                let e = expected
                    .iter()
                    .take(10)
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(", ");
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Unexpected token")
                    .with_label(Label::new((info.id()?, info.span())).with_message(
                        &if expected.len() > 10 {
                            format!(r#"Unexpected token "{found}", expected: {e}, ..."#)
                        } else {
                            format!(r#"Unexpected token "{found}", expected: {e}"#)
                        },
                    ))
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::DuplicateMetaKey { info, key } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Duplicate meta key")
                    .with_label(
                        Label::new((info.id()?, info.span()))
                            .with_message(&format!("Duplicate meta key `{}`", key)),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::GenericWithArgs { info, name } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Generic type with arguments")
                    .with_label(
                        Label::new((info.id()?, info.span()))
                            .with_message(&format!("Generic type `{}` with arguments", name)),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::UnresolvedTypeName { info, name } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Unresolved type name")
                    .with_label(
                        Label::new((info.id()?, info.span()))
                            .with_message(&format!("Unresolved type name `{}`", name)),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::ExpectedVariant { info } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Expected variant")
                    .with_label(
                        Label::new((info.id()?, info.span())).with_message("Expected variant"),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::ExpectedPlaceExpr { info } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Expected place expression")
                    .with_label(
                        Label::new((info.id()?, info.span())).with_message(
                            "Expected place expression, e.g., `x` or `x.y` or `x[0]`",
                        ),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::UnexpectedTypeArgs { info } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Unexpected type arguments")
                    .with_label(
                        Label::new((info.id()?, info.span()))
                            .with_message("Unexpected type arguments"),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::ExpectedVar { info } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Expected variable")
                    .with_label(
                        Label::new((info.id()?, info.span())).with_message("Expected variable"),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::TypeMismatch { lhs, rhs, info } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Type mismatch")
                    .with_label(
                        Label::new((info.id()?, info.span()))
                            .with_message(&format!("Expected `{}`, found `{}`", lhs, rhs)),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::NonExhaustiveMatch { info: _ } => todo!(),
            Error::NameClash { info0, info1 } => {
                Report::build(ReportKind::Error, info0.id()?, info0.span().start)
                    .with_message("Name clash")
                    .with_label(
                        Label::new((info0.id()?, info0.span()))
                            .with_message("Name already defined in this scope."),
                    )
                    .with_label(Label::new((info1.id()?, info1.span())).with_message("Here."))
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::DuplicateVariant { info } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Duplicate variant")
                    .with_label(
                        Label::new((info.id()?, info.span())).with_message("Duplicate variant"),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::BreakOutsideLoop { info } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Break outside loop")
                    .with_label(
                        Label::new((info.id()?, info.span())).with_message("Break outside loop"),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::BreakOutsideInfiniteLoop { info } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Break with argument outside infinite loop")
                    .with_label(
                        Label::new((info.id()?, info.span()))
                            .with_message("Break with argument outside infinite loop"),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::ContinueOutsideLoop { info } => todo!(),
            Error::DuplicateRow { info: _ } => todo!(),
            Error::TypeInValuePosition { info: _ } => todo!(),
            Error::ValueInTypePosition { info: _ } => todo!(),
            Error::ExpectedFunction { info: _ } => todo!(),
            Error::UnresolvedName { info } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Unresolved name")
                    .with_label(
                        Label::new((info.id()?, info.span())).with_message("Unresolved name"),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::GenericError { info, text } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Generic error")
                    .with_label(Label::new((info.id()?, info.span())).with_message(text))
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::InfiniteType { info, t } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Infinite type")
                    .with_label(
                        Label::new((info.id()?, info.span()))
                            .with_message(&format!("Infinite type: `{}`", t)),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::WrongNumberOfTypeArgs {
                name,
                expected,
                found,
                info0,
                info1,
            } => {
                Report::build(ReportKind::Error, info0.id()?, info0.span().start)
                    .with_message("Wrong number of type arguments")
                    .with_label(
                        Label::new((info0.id()?, info0.span())).with_message(&format!(
                            "`{}` expected `{}`, found `{}`",
                            name, expected, found
                        )),
                    )
                    .with_label(Label::new((info1.id()?, info1.span())).with_message("Here."))
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::UncompileableCode { info, msg } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Found un-compileable code")
                    .with_label(
                        Label::new((info.id()?, info.span()))
                            .with_message(format!("This code cannot (yet) be compiled. {msg}.")),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::ExpectedVarOrVal { info } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Expected `var` or `val`")
                    .with_label(
                        Label::new((info.id()?, info.span()))
                            .with_message("Expected `var` or `val`, found this."),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::NameNotFound { info, name } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Name not found")
                    .with_label(
                        Label::new((info.id()?, info.span()))
                            .with_message(&format!("Name `{}` not found", name)),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::ExpectedIrrefutablePattern { info } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Expected irrefutable pattern")
                    .with_label(
                        Label::new((info.id()?, info.span()))
                            .with_message("Expected irrefutable pattern"),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::RowNotFound { info, x } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Row not found")
                    .with_label(
                        Label::new((info.id()?, info.span()))
                            .with_message(&format!("Row `{}` not found", x)),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::InterpreterError { info, s } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Interpreter error")
                    .with_label(
                        Label::new((info.id()?, info.span()))
                            .with_message(&format!("Interpreter error: `{}`", s)),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::WrongNumberOfArguments {
                info,
                expected,
                found,
            } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Wrong number of arguments")
                    .with_label(
                        Label::new((info.id()?, info.span()))
                            .with_message(&format!("Expected {}, found {}", expected, found)),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
            Error::SynError { info, e } => {
                Report::build(ReportKind::Error, info.id()?, info.span().start)
                    .with_message("Syntax error")
                    .with_label(
                        Label::new((info.id()?, info.span())).with_message(&format!("{}", e)),
                    )
                    .finish()
                    .eprint(sources)
                    .ok()?;
            }
        };
        Some(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LexerError {
    ParseBoolError(std::str::ParseBoolError),
    ParseCharError(std::char::ParseCharError),
    ParseFloatError(std::num::ParseFloatError),
    ParseIntError(std::num::ParseIntError),
    InvalidToken,
}

impl Default for LexerError {
    fn default() -> Self {
        Self::InvalidToken
    }
}

impl From<std::str::ParseBoolError> for LexerError {
    fn from(err: std::str::ParseBoolError) -> Self {
        Self::ParseBoolError(err)
    }
}

impl From<std::char::ParseCharError> for LexerError {
    fn from(err: std::char::ParseCharError) -> Self {
        Self::ParseCharError(err)
    }
}

impl From<std::num::ParseFloatError> for LexerError {
    fn from(err: std::num::ParseFloatError) -> Self {
        Self::ParseFloatError(err)
    }
}

impl From<std::num::ParseIntError> for LexerError {
    fn from(err: std::num::ParseIntError) -> Self {
        Self::ParseIntError(err)
    }
}

impl Display for LexerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseBoolError(err) => write!(f, "Invalid token: {}", err),
            Self::ParseCharError(err) => write!(f, "Invalid token: {}", err),
            Self::ParseFloatError(err) => write!(f, "Invalid token: {}", err),
            Self::ParseIntError(err) => write!(f, "Invalid token: {}", err),
            Self::InvalidToken => write!(f, "Invalid token"),
        }
    }
}
