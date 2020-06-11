use {
    crate::{ast::*, error::*},
    codespan::Span,
    grammar::*,
    lalrpop_util::lalrpop_mod,
    num_traits::Num,
    smol_str::SmolStr,
    std::{fmt::Display, str::FromStr},
};

lalrpop_mod!(pub grammar);

pub type ErrorRecovery<'i> = lalrpop_util::ErrorRecovery<usize, Token<'i>, CompilerError>;
pub type ParseError<'i> = lalrpop_util::ParseError<usize, Token<'i>, CompilerError>;

pub struct Parser<'i> {
    errors: Vec<ErrorRecovery<'i>>,
}

impl<'i> Parser<'i> {
    pub fn new() -> Parser<'i> {
        let errors = Vec::new();
        Parser { errors }
    }

    pub fn parse(&mut self, source: &'i str) -> Script {
        ScriptParser::new().parse(&mut self.errors, source).unwrap()
    }

    pub fn errors(self) -> Vec<CompilerError> {
        self.errors
            .into_iter()
            .map(|recovery| recovery.error)
            .map(Into::into)
            .collect()
    }
}

impl<'i> From<ParseError<'i>> for CompilerError {
    fn from(error: ParseError) -> CompilerError {
        match error {
            ParseError::User { error } => error,
            ParseError::ExtraToken { token: (l, t, r) } => CompilerError::ExtraToken {
                found: SmolStr::from(t.1),
                span: Span::from(l as u32..r as u32),
            },
            ParseError::InvalidToken { location: l } => CompilerError::InvalidToken {
                span: Span::from(l as u32..l as u32),
            },
            ParseError::UnrecognizedEOF {
                location: l,
                expected,
            } => CompilerError::UnrecognizedEOF {
                span: Span::from(l as u32..l as u32),
                expected,
            },
            ParseError::UnrecognizedToken {
                token: (l, t, r),
                expected,
            } => CompilerError::UnrecognizedToken {
                found: SmolStr::from(t.1),
                span: Span::from(l as u32..r as u32),
                expected,
            },
        }
    }
}

pub fn parse_lit<'i, 'e, T: FromStr>(
    Spanned(l, s, r): Spanned<&'i str>,
    errors: &'e mut Vec<ErrorRecovery>,
) -> Option<T>
where
    <T as FromStr>::Err: Display,
{
    match T::from_str(s) {
        Err(error) => {
            let span = Span::new(l as u32, r as u32);
            let msg = error.to_string();
            let error = CompilerError::BadLiteral { msg, span }.into();
            errors.push(error);
            None
        }
        Ok(l) => Some(l),
    }
}

pub fn parse_lit_radix<'i, 'e, T: Num>(
    Spanned(l, s, r): Spanned<&'i str>,
    radix: u32,
    prefix: usize,
    suffix: usize,
    errors: &'e mut Vec<ErrorRecovery>,
) -> Option<T>
where
    <T as Num>::FromStrRadixErr: Display,
{
    match T::from_str_radix(&s[prefix..s.len() - suffix], radix) {
        Err(error) => {
            let span = Span::new(l as u32, r as u32);
            let msg = error.to_string();
            let error = CompilerError::BadLiteral { msg, span }.into();
            errors.push(error);
            None
        }
        Ok(l) => Some(l),
    }
}

impl<'i> From<CompilerError> for ErrorRecovery<'i> {
    fn from(error: CompilerError) -> ErrorRecovery<'i> {
        ErrorRecovery {
            error: ParseError::User { error },
            dropped_tokens: vec![],
        }
    }
}
