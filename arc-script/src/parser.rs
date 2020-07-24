use {
    crate::{ast::*, error::*, info::Info, symbols::*},
    codespan::Span,
    grammar::*,
    lalrpop_util::lalrpop_mod,
    num_traits::Num,
    regex::Regex,
    smol_str::SmolStr,
    std::{fmt::Display, str::FromStr},
};

lalrpop_mod!(#[allow(clippy::all)] pub grammar);

pub type ErrorRecovery<'i> = lalrpop_util::ErrorRecovery<usize, Token<'i>, CompilerError>;
pub type ParseError<'i> = lalrpop_util::ParseError<usize, Token<'i>, CompilerError>;

impl Script<'_> {
    pub fn parse(source: &str) -> Script<'_> {
        let mut errors = Vec::new();
        let mut table = SymbolTable::new();
        let mut stack = SymbolStack::new();
        let (tydefs, fundefs, body) = ScriptParser::new()
            .parse(&mut errors, &mut stack, &mut table, source)
            .unwrap();
        let errors = errors
            .into_iter()
            .map(|recovery| recovery.error)
            .map(Into::into)
            .collect();
        let info = Info::new(table, errors, source);
        Script {
            tydefs,
            fundefs,
            body,
            info,
        }
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

pub fn parse_lit<'i, T: FromStr>(Spanned(l, s, r): Spanned<&'i str>) -> CompilerResult<T>
where
    <T as FromStr>::Err: Display,
{
    match T::from_str(s) {
        Err(error) => {
            let span = Span::new(l as u32, r as u32);
            let msg = error.to_string();
            Err(CompilerError::BadLiteral { msg, span })
        }
        Ok(l) => Ok(l),
    }
}

pub fn parse_lit_radix<'i, T: Num>(
    Spanned(l, s, r): Spanned<&'i str>,
    radix: u32,
    prefix: usize,
    suffix: usize,
) -> CompilerResult<T>
where
    <T as Num>::FromStrRadixErr: Display,
{
    match T::from_str_radix(&s[prefix..s.len() - suffix], radix) {
        Err(error) => {
            let span = Span::new(l as u32, r as u32);
            let msg = error.to_string();
            Err(CompilerError::BadLiteral { msg, span })
        }
        Ok(ok) => Ok(ok),
    }
}

// Parses a r"foo" string into a regex::Regex
pub fn parse_regex<'i>(Spanned(l, s, r): Spanned<&'i str>) -> CompilerResult<Regex> {
    match Regex::from_str(&s[1..s.len()]) {
        Err(error) => {
            let span = Span::new(l as u32, r as u32);
            let msg = error.to_string();
            Err(CompilerError::BadLiteral { msg, span })
        }
        Ok(ok) => Ok(ok),
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