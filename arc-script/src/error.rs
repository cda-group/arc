use {
    codespan::Span,
    codespan_reporting::{
        diagnostic::{self, Label},
        files,
        term::{self, termcolor::*},
    },
    smol_str::SmolStr,
};

pub type Error = Box<dyn std::error::Error + 'static>;

pub type SimpleFile<'i> = files::SimpleFile<&'i str, &'i str>;
pub type Diagnostic = diagnostic::Diagnostic<()>;
pub type CompilerResult<T> = Result<T, CompilerError>;

pub struct Reporter<'i> {
    pub file: SimpleFile<'i>,
    pub diags: Vec<Diagnostic>,
}

impl<'i> Reporter<'i> {
    pub fn new(source: &'i str, errors: Vec<impl Into<Diagnostic>>) -> Reporter<'i> {
        let file = SimpleFile::new("input", source);
        let diags = errors.into_iter().map(Into::into).collect();
        Reporter { file, diags }
    }

    pub fn emit(self) {
        let writer = StandardStream::stderr(ColorChoice::Always);
        let writer = &mut writer.lock();
        let config = codespan_reporting::term::Config::default();
        for diag in &self.diags {
            term::emit(writer, &config, &self.file, diag).unwrap();
        }
    }

    pub fn emit_as_str(self) -> String {
        let mut writer = codespan_reporting::term::termcolor::Buffer::ansi();
        let config = codespan_reporting::term::Config::default();
        for diag in &self.diags {
            term::emit(&mut writer, &config, &self.file, diag).unwrap();
        }
        format!(
            "\n{}",
            std::str::from_utf8(&mut writer.into_inner())
                .unwrap()
                .to_owned()
        )
    }
}

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
        found: SmolStr,
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
        found: SmolStr,
        span: Span,
        expected: Vec<String>,
    },
    TypeMismatch {
        lhs: String,
        rhs: String,
        span: Span,
    },
    VarNotFound {
        name: SmolStr,
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
}

impl From<CompilerError> for Diagnostic {
    fn from(error: CompilerError) -> Diagnostic {
        match error {
            CompilerError::BadLiteral { msg, span } => Diagnostic::error()
                .with_message("Bad literal")
                .with_labels(vec![Label::primary((), span).with_message(msg)]),
            CompilerError::BadUri { msg, span } => Diagnostic::error()
                .with_message("Bad URI")
                .with_labels(vec![Label::primary((), span).with_message(msg)]),
            CompilerError::ExtraToken { found, span } => Diagnostic::error()
                .with_message(format!("Extraneous token {}", found))
                .with_labels(vec![Label::primary((), span)]),
            CompilerError::InvalidToken { span } => Diagnostic::error()
                .with_message("Invalid token")
                .with_labels(vec![Label::primary((), span)]),
            CompilerError::UnrecognizedEOF { span, expected } => Diagnostic::error()
                .with_message("Unrecognized end of file")
                .with_labels(vec![Label::primary((), span)
                    .with_message(format!("expected {}", expected.join(", ")))]),
            CompilerError::UnrecognizedToken {
                found,
                expected,
                span,
            } => Diagnostic::error()
                .with_message(format!("Unrecognized token {}", found))
                .with_labels(vec![Label::primary((), span)
                    .with_message(format!("expected {}", expected.join(", ")))]),
            CompilerError::TypeMismatch { lhs, rhs, span } => Diagnostic::error()
                .with_message(format!("Type mismatch"))
                .with_labels(vec![
                    Label::primary((), span).with_message(format!("{} != {}", lhs, rhs))
                ]),
            CompilerError::VarNotFound { name, span } => Diagnostic::error()
                .with_message(format!("Identifier `{}` not bound to anything", name))
                .with_labels(vec![Label::primary((), span).with_message("Used here")]),
            CompilerError::DisallowedDimExpr { span } => Diagnostic::error()
                .with_message("Disallowed expression in dimension")
                .with_labels(vec![Label::primary((), span).with_message("Found here")]),
            CompilerError::ShapeUnsat => Diagnostic::error().with_message("Unsatisfiable shape"),
            CompilerError::ShapeUnknown => Diagnostic::error().with_message("Unknown shape"),
            CompilerError::NonExhaustiveMatch { span } => Diagnostic::error()
                .with_message("Match is non-exhaustive")
                .with_labels(vec![Label::primary((), span).with_message("Missing cases")]),
        }
    }
}
