use crate::compiler::ast;
use crate::compiler::ast::from::lexer::Token;
use crate::compiler::info::diags::{Diagnostic, Error};
use crate::compiler::info::files::{ByteIndex, FileId, Loc};

/// Dropped tokens + errors produced while parsing with LALRPOP.
pub(crate) type ErrorRecovery<'i> = lalrpop_util::ErrorRecovery<ByteIndex, Token, ()>;

/// Errors produced while parsing with LALRPOP.
pub(crate) type ParseError<'i> = lalrpop_util::ParseError<ByteIndex, Token, ()>;

impl Diagnostic {
    /// Converts an LALRPOP `ErrorRecovery` into a `Diagnostic`.
    pub(crate) fn from(recovery: ErrorRecovery, file: FileId) -> Diagnostic {
        match recovery.error {
            ParseError::User { .. } => unreachable!(),
            ParseError::ExtraToken { token: (l, t, r) } => Error::ExtraToken {
                found: t,
                loc: Loc::from_range(file, l..r).into(),
            }
            .into(),
            ParseError::InvalidToken { location: l } => Error::InvalidToken {
                loc: Loc::from_range(file, l..l).into(),
            }
            .into(),
            ParseError::UnrecognizedEOF {
                location: l,
                expected,
            } => Error::UnrecognizedEOF {
                loc: Loc::from_range(file, l..l).into(),
                expected,
            }
            .into(),
            ParseError::UnrecognizedToken {
                token: (l, t, r),
                expected,
            } => Error::UnrecognizedToken {
                found: t,
                loc: Loc::from_range(file, l..r).into(),
                expected,
            }
            .into(),
        }
    }
}
