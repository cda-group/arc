use {
    crate::{ast::*, info::*, pretty::*},
    codespan::Span,
    codespan_reporting::{
        diagnostic::{self, Label},
        files,
        term::{
            self,
            termcolor::{Buffer, ColorChoice, StandardStream, WriteColor},
            Config,
        },
    },
    std::io,
    std::str,
};

pub type SimpleFile<'i> = files::SimpleFile<&'i str, &'i str>;
pub type Diagnostic = diagnostic::Diagnostic<()>;
pub type CompilerResult<T> = Result<T, CompilerError>;

impl<'i> Script<'i> {
    fn emit<W>(&self, writer: &mut W) -> io::Result<()>
    where
        W: io::Write + WriteColor,
    {
        let file = &SimpleFile::new("input", self.info.source);
        let config = &Config::default();
        self.info
            .errors
            .iter()
            .map(|error| error.to_diagnostic(&self.info))
            .try_for_each(|diag| term::emit(writer, config, file, &diag))
    }

    pub fn emit_to_stdout(&self) {
        let writer = StandardStream::stderr(ColorChoice::Never);
        let writer = &mut writer.lock();
        self.emit(writer).unwrap()
    }

    pub fn emit_as_str(&self) -> String {
        let mut writer = Buffer::ansi();
        self.emit(&mut writer).unwrap();
        let writer = writer.into_inner();
        format!("{}", str::from_utf8(&writer).unwrap().to_owned())
    }
}

/// Represents the various errors reported by the compiler.
#[derive(Debug)]
pub enum CompilerError {
    BadLiteral {
        msg: String,
        span: Span,
    },
    BadUri {
        msg: String,
        span: Span,
    },
    ExtraToken {
        found: String,
        span: Span,
    },
    InvalidToken {
        span: Span,
    },
    UnrecognizedEOF {
        span: Span,
        expected: Vec<String>,
    },
    UnrecognizedToken {
        found: String,
        span: Span,
        expected: Vec<String>,
    },
    TypeMismatch {
        lhs: Type,
        rhs: Type,
        span: Span,
    },
    VarNotFound {
        name: Ident,
        span: Span,
    },
    DisallowedDimExpr {
        span: Span,
    },
    ShapeUnknown,
    ShapeUnsat,
    NonExhaustiveMatch {
        span: Span,
    },
    NameClash,
}

/// Converts a compiler error into a diagnostic which can be emitted by codespan.
impl CompilerError {
    pub fn to_diagnostic(&self, info: &Info) -> Diagnostic {
        match self {
            CompilerError::BadLiteral { msg, span } => Diagnostic::error()
                .with_message("Bad literal")
                .with_labels(vec![Label::primary((), *span).with_message(msg)]),
            CompilerError::BadUri { msg, span } => Diagnostic::error()
                .with_message("Bad URI")
                .with_labels(vec![Label::primary((), *span).with_message(msg)]),
            CompilerError::ExtraToken { found, span } => Diagnostic::error()
                .with_message(format!("Extraneous token {}", found))
                .with_labels(vec![Label::primary((), *span)]),
            CompilerError::InvalidToken { span } => Diagnostic::error()
                .with_message("Invalid token")
                .with_labels(vec![Label::primary((), *span)]),
            CompilerError::UnrecognizedEOF { span, expected } => Diagnostic::error()
                .with_message("Unrecognized end of file")
                .with_labels(vec![Label::primary((), *span)
                    .with_message(format!("expected {}", expected.join(", ")))]),
            CompilerError::UnrecognizedToken {
                found,
                expected,
                span,
            } => Diagnostic::error()
                .with_message(format!("Unrecognized token {}", found))
                .with_labels(vec![Label::primary((), *span)
                    .with_message(format!("expected {}", expected.join(", ")))]),
            CompilerError::TypeMismatch { lhs, rhs, span } => Diagnostic::error()
                .with_message("Type mismatch")
                .with_labels(vec![Label::primary((), *span).with_message(format!(
                    "{} != {}",
                    lhs.brief(info),
                    rhs.brief(info)
                ))]),
            CompilerError::VarNotFound { name, span } => Diagnostic::error()
                .with_message(format!(
                    "Identifier `{}` not bound to anything",
                    info.table.get_decl_name(name)
                ))
                .with_labels(vec![Label::primary((), *span).with_message("Used here")]),
            CompilerError::DisallowedDimExpr { span } => Diagnostic::error()
                .with_message("Disallowed expression in dimension")
                .with_labels(vec![Label::primary((), *span).with_message("Found here")]),
            CompilerError::ShapeUnsat => Diagnostic::error().with_message("Unsatisfiable shape"),
            CompilerError::ShapeUnknown => Diagnostic::error().with_message("Unknown shape"),
            CompilerError::NonExhaustiveMatch { span } => Diagnostic::error()
                .with_message("Match is non-exhaustive")
                .with_labels(vec![Label::primary((), *span).with_message("Missing cases")]),
            CompilerError::NameClash => Diagnostic::error().with_message("Name clash"),
        }
    }
}
