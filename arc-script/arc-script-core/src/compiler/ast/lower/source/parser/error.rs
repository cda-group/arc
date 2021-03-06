//! Module for converting errors emitted by `LALRPOP` into compiler diagnostics.

use crate::compiler::ast::lower::source::lexer::sem_tokens::Token;
use crate::compiler::info::diags::Diagnostic;
use crate::compiler::info::diags::Error;
use crate::compiler::info::files::ByteIndex;
use crate::compiler::info::files::FileId;
use crate::compiler::info::files::Loc;

/// Dropped tokens + errors produced while parsing with LALRPOP.
pub(crate) type ErrorRecovery = lalrpop_util::ErrorRecovery<ByteIndex, Token, ()>;

/// Errors produced while parsing with LALRPOP.
pub(crate) type ParseError = lalrpop_util::ParseError<ByteIndex, Token, ()>;

impl Diagnostic {
    /// Converts an LALRPOP `ErrorRecovery` into a `Diagnostic`.
    pub(crate) fn from(recovery: ErrorRecovery, file: FileId) -> impl Into<Self> {
        match recovery.error {
            /// User errors (lexer errors) are handled by the lexer.
            ParseError::User { error: () } => unreachable!(),
            /// Error generated by the parser when it encounters additional, unexpected tokens.
            ParseError::ExtraToken { token: (l, t, r) } => Error::ExtraToken {
                found: t,
                loc: Loc::from_range(file, l..r),
            },
            /// Error generated by the parser when it encounters a token (or EOF) it did not expect.
            ParseError::InvalidToken { location: l } => Error::InvalidToken {
                loc: Loc::from_range(file, l..l),
            },
            /// Error generated by the parser when it encounters an EOF it did not expect.
            ParseError::UnrecognizedEOF {
                location: l,
                expected,
            } => Error::UnrecognizedEOF {
                loc: Loc::from_range(file, l..l),
                expected,
            },
            /// Error generated by the parser when it encounters a token it did not expect.
            ParseError::UnrecognizedToken {
                token: (l, t, r),
                expected,
            } => Error::UnrecognizedToken {
                found: t,
                loc: Loc::from_range(file, l..r),
                expected,
            },
        }
    }
}
