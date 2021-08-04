#![allow(unused)]

/// Module for displaying tokens.
mod display;
/// Module for defining the format of numeric literals.
mod numfmt;
/// Module which contains raw logos tokens.
mod raw_tokens;
/// Module which contains semantic tokens.
pub(crate) mod sem_tokens;

use crate::compiler::ast::lower::source::lexer::raw_tokens::LogosToken;
use crate::compiler::ast::lower::source::lexer::sem_tokens::Token;
use crate::compiler::info::diags::DiagInterner;
use crate::compiler::info::diags::Error;
use crate::compiler::info::diags::Result;
use crate::compiler::info::files::ByteIndex;
use crate::compiler::info::files::FileId;
use crate::compiler::info::files::Loc;
use crate::compiler::info::files::Span;
use crate::compiler::info::names::NameId;
use crate::compiler::info::names::NameInterner;

use arc_script_core_shared::New;

use lexical_core::parse_format;
use lexical_core::NumberFormat;
use logos::Lexer as LogosLexer;
use logos::Logos;
use time::Duration;
use time::PrimitiveDateTime as DateTime;

use std::convert::TryFrom;

/// Number of tabs used in semantic indentation.
pub(crate) const TABSIZE: usize = 4;

/// A token for used for representing semantic indentation.
#[derive(Debug, PartialEq)]
pub(crate) enum Scope {
    /// Indentation level has dropped by N since last token.
    Dedent(usize),
    /// Indentation level has increased by 1 since last token.
    Indent,
    /// Indenation level is unchanged since last token.
    Unchanged,
}

/// A lexer for scanning Arc-Script semantic tokens.
pub(crate) struct Lexer<'i> {
    /// Underlying generated lexer DFA for scanning raw tokens.
    logos: LogosLexer<'i, LogosToken>,
    /// Number of levels to un-indent.
    dedents: usize,
    /// Current indentation.
    indents: usize,
    /// Format used by [`lexical_core`] for parsing numeric literals.
    numfmt: NumberFormat,
    /// Map which stores all symbolic names encountered during lexing.
    names: &'i mut NameInterner,
    /// Identifier of the file which the scanned source code belongs to.
    file: FileId,
    /// Map which stores diagnostics encountered during scanning.
    pub(crate) diags: DiagInterner,
}
impl<'i> Lexer<'i> {
    /// Returns a new lexer with an initial state.
    pub(crate) fn new(source: &'i str, file: FileId, names: &'i mut NameInterner) -> Self {
        Self {
            logos: LogosToken::lexer(source),
            dedents: 0,
            indents: 0,
            numfmt: numfmt::compile(),
            diags: DiagInterner::default(),
            file,
            names,
        }
    }

    /// Scans semantic indentation.
    #[inline]
    fn newline(&mut self) -> Result<Scope> {
        // Offset to the last newline
        let offset = self
            .logos
            .slice()
            .chars()
            .enumerate()
            .filter(|(_, c)| *c == '\n')
            .last()
            .unwrap()
            .0;
        let len = self.logos.span().len() + offset - 1;
        if len % TABSIZE == 0 {
            let old = self.indents;
            self.indents = len / TABSIZE;
            match self.indents {
                new if new < old => Ok(Scope::Dedent(old - new)),
                new if new == old + 1 => Ok(Scope::Indent),
                new if new == old => Ok(Scope::Unchanged),
                _ => Err(Error::TooMuchIndent { loc: self.loc() }.into()),
            }
        } else {
            Err(Error::BadIndent { loc: self.loc() }.into())
        }
    }

    /// Scans a numeric literal using `lexical_core`.
    fn lit<N: lexical_core::FromLexicalFormat>(
        &mut self,
        prefix: usize,
        suffix: usize,
    ) -> Result<N> {
        parse_format(self.trim(prefix, suffix).as_bytes(), self.numfmt).map_err(|msg| {
            Error::LexicalCore {
                loc: self.loc(),
                err: msg,
            }
            .into()
        })
    }

    /// Scans a `DateTime` literal using `time`.
    fn datetime(&mut self, format: &str) -> Result<DateTime> {
        DateTime::parse(self.logos.slice(), format).map_err(|msg| {
            Error::Time {
                loc: self.loc(),
                err: msg,
            }
            .into()
        })
    }

    /// Returns the current location of the lexer.
    #[inline]
    fn loc(&self) -> Loc {
        let file = self.file;
        let span = self.span();
        Loc::Real(file, span)
    }

    /// Returns the current span of the lexer.
    #[inline]
    fn span(&self) -> Span {
        let span = self.logos.span();
        let start = ByteIndex::try_from(span.start).unwrap();
        let end = ByteIndex::try_from(span.end).unwrap();
        Span::new(start, end)
    }

    /// Trims the lexer's current slice by stripping its prefix and suffix. Used to for example
    /// strip quotes from strings.
    fn trim(&self, prefix: usize, suffix: usize) -> &str {
        &self.logos.slice()[prefix..self.logos.span().len() - suffix]
    }

    /// Returns the next token or an error.
    #[rustfmt::skip]
    fn token(&mut self) -> Result<Option<Token>> {
        let slice = self.logos.slice();
//         if self.dedents > 0 {
//             self.dedents -= 1;
//             return Ok(Some(Token::Dedent));
//         }
        while let Some(token) = self.logos.next() {
            let token = match token {
                LogosToken::Error      => return Err(Error::InvalidToken { loc: self.loc() }.into()),
                LogosToken::Comment | LogosToken::Newline => continue,
//                 match self.newline()? {
//                     Scope::Dedent(dedents) => {
//                         self.dedents = dedents - 1;
//                         Token::Dedent
//                     }
//                     Scope::Indent => Token::Indent,
//                     Scope::Unchanged => continue,
//                 },
//=============================================================================
// Grouping
//=============================================================================
                LogosToken::BraceL     => Token::BraceL,
                LogosToken::BraceR     => Token::BraceR,
                LogosToken::BraceLR    => Token::BraceLR,
                LogosToken::BrackL     => Token::BrackL,
                LogosToken::BrackR     => Token::BrackR,
                LogosToken::BrackLR    => Token::BrackLR,
                LogosToken::ParenL     => Token::ParenL,
                LogosToken::ParenR     => Token::ParenR,
                LogosToken::ParenLR    => Token::ParenLR,
                LogosToken::AngleL     => Token::AngleL,
                LogosToken::AngleR     => Token::AngleR,
                LogosToken::AngleLR    => Token::AngleLR,
//=============================================================================
// Operators
//=============================================================================
                LogosToken::ArrowR     => Token::ArrowR,
                LogosToken::Bar        => Token::Bar,
                LogosToken::BarBar     => Token::BarBar,
                LogosToken::Colon      => Token::Colon,
                LogosToken::ColonColon => Token::ColonColon,
                LogosToken::Comma      => Token::Comma,
                LogosToken::Dot        => Token::Dot,
                LogosToken::DotDot     => Token::DotDot,
                LogosToken::DotDotEq   => Token::DotDotEq,
                LogosToken::Equ        => Token::Equ,
                LogosToken::EquEqu     => Token::EquEqu,
                LogosToken::Geq        => Token::Geq,
                LogosToken::AtSign     => Token::AtSign,
                LogosToken::Imply      => Token::Imply,
                LogosToken::Leq        => Token::Leq,
                LogosToken::Minus      => Token::Minus,
                LogosToken::Neq        => Token::Neq,
                LogosToken::Percent    => Token::Percent,
                LogosToken::Plus       => Token::Plus,
                LogosToken::Semi       => Token::Semi,
                LogosToken::Slash      => Token::Slash,
                LogosToken::Star       => Token::Star,
                LogosToken::StarStar   => Token::StarStar,
                LogosToken::Tilde      => Token::Tilde,
                LogosToken::Underscore => Token::Underscore,
//=============================================================================
// Unused Operators
//=============================================================================
                // LogosToken::Amp        => Token::Amp,
                // LogosToken::AmpAmp     => Token::AmpAmp,
                // LogosToken::ArrowL     => Token::ArrowL,
                // LogosToken::Bang       => Token::Bang,
                // LogosToken::Caret      => Token::Caret,
                // LogosToken::Dollar     => Token::Dollar,
                // LogosToken::Gt         => Token::Gt,
                // LogosToken::Lt         => Token::Lt,
                // LogosToken::LtGt       => Token::LtGt,
                // LogosToken::Qm         => Token::Qm,
                // LogosToken::QmQmQm     => Token::QmQmQm,
                // LogosToken::SemiSemi   => Token::SemiSemi,
//=============================================================================
// Keywords
//=============================================================================
                LogosToken::After      => Token::After,
                LogosToken::And        => Token::And,
                LogosToken::As         => Token::As,
                LogosToken::Band       => Token::Band,
                LogosToken::Bor        => Token::Bor,
                LogosToken::Bxor       => Token::Bxor,
                LogosToken::Break      => Token::Break,
                LogosToken::By         => Token::By,
                LogosToken::Crate      => Token::Crate,
                LogosToken::Continue   => Token::Continue,
                LogosToken::Else       => Token::Else,
                LogosToken::Emit       => Token::Emit,
                LogosToken::Enum       => Token::Enum,
                LogosToken::Every      => Token::Every,
                LogosToken::Extern     => Token::Extern,
                LogosToken::For        => Token::For,
                LogosToken::Fun        => Token::Fun,
                LogosToken::If         => Token::If,
                LogosToken::In         => Token::In,
                LogosToken::Is         => Token::Is,
                LogosToken::Let        => Token::Let,
                LogosToken::Log        => Token::Log,
                LogosToken::Loop       => Token::Loop,
                LogosToken::Match      => Token::Match,
                LogosToken::Not        => Token::Not,
                LogosToken::On         => Token::On,
                LogosToken::Or         => Token::Or,
                LogosToken::Return     => Token::Return,
                LogosToken::Task       => Token::Task,
                LogosToken::Type       => Token::Type,
                LogosToken::Val        => Token::Val,
                LogosToken::Var        => Token::Var,
                LogosToken::Unwrap     => Token::Unwrap,
                LogosToken::Enwrap     => Token::Enwrap,
                LogosToken::Use        => Token::Use,
                LogosToken::Xor        => Token::Xor,
//=============================================================================
// Unused Keywords
//=============================================================================
                // LogosToken::Add        => Token::Add,
                // LogosToken::Box        => Token::Box,
                // LogosToken::Do         => Token::Do,
                // LogosToken::End        => Token::End,
                // LogosToken::Of         => Token::Of,
                // LogosToken::Port       => Token::Port,
                // LogosToken::Pub        => Token::Pub,
                // LogosToken::Reduce     => Token::Reduce,
                // LogosToken::Shutdown   => Token::Shutdown,
                // LogosToken::Sink       => Token::Sink,
                // LogosToken::Source     => Token::Source,
                // LogosToken::Startup    => Token::Startup,
                // LogosToken::State      => Token::State,
                // LogosToken::Then       => Token::Then,
                // LogosToken::Timeout    => Token::Timeout,
                // LogosToken::Timer      => Token::Timer,
                // LogosToken::Trigger    => Token::Trigger,
                // LogosToken::Where      => Token::Where,
//=============================================================================
// Primitive Types
//=============================================================================
                LogosToken::Bool       => Token::Bool,
                LogosToken::F32        => Token::F32,
                LogosToken::F64        => Token::F64,
                LogosToken::I8         => Token::I8,
                LogosToken::I16        => Token::I16,
                LogosToken::I32        => Token::I32,
                LogosToken::I64        => Token::I64,
                LogosToken::U8         => Token::U8,
                LogosToken::U16        => Token::U16,
                LogosToken::U32        => Token::U32,
                LogosToken::U64        => Token::U64,
                LogosToken::Str        => Token::Str,
                LogosToken::Unit       => Token::Unit,
                LogosToken::Char       => Token::Char,
                LogosToken::Size       => Token::Size,
//=============================================================================
// Identifiers and Literals
//=============================================================================
                LogosToken::LitI8           => Token::LitI8(self.lit(0, 2)?),
                LogosToken::LitI16          => Token::LitI16(self.lit(0, 3)?),
                LogosToken::LitI32          => Token::LitI32(self.lit(0, 0)?),
                LogosToken::LitI64          => Token::LitI64(self.lit(0, 3)?),
                LogosToken::LitU8           => Token::LitU8(self.lit(0, 2)?),
                LogosToken::LitU16          => Token::LitU16(self.lit(0, 3)?),
                LogosToken::LitU32          => Token::LitU32(self.lit(0, 3)?),
                LogosToken::LitU64          => Token::LitU64(self.lit(0, 3)?),
                LogosToken::LitF32          => Token::LitF32(self.lit(0, 3)?),
                LogosToken::LitF64          => Token::LitF64(self.lit(0, 0)?),
                LogosToken::LitTrue         => Token::LitBool(true),
                LogosToken::LitFalse        => Token::LitBool(false),
                LogosToken::LitChar         => Token::LitChar(self.trim(1, 1).chars().next().unwrap()),
                LogosToken::LitStr          => Token::LitStr(self.trim(1, 1).to_string()),
                LogosToken::LitDate         => Token::LitDateTime(self.datetime("%F")?),
                LogosToken::LitDateTime     => Token::LitDateTime(self.datetime("%FT%T")?),
                LogosToken::LitDateTimeZone => Token::LitDateTime(self.datetime("%FT%T%Z")?),
                LogosToken::LitDurationNs   => Token::LitDuration(Duration::seconds(self.lit(0, 2)?)),
                LogosToken::LitDurationUs   => Token::LitDuration(Duration::microseconds(self.lit(0, 2)?)),
                LogosToken::LitDurationMs   => Token::LitDuration(Duration::milliseconds(self.lit(0, 2)?)),
                LogosToken::LitDurationS    => Token::LitDuration(Duration::nanoseconds(self.lit(0, 1)?)),
                LogosToken::LitDurationM    => Token::LitDuration(Duration::minutes(self.lit(0, 1)?)),
                LogosToken::LitDurationH    => Token::LitDuration(Duration::hours(self.lit(0, 1)?)),
                LogosToken::LitDurationD    => Token::LitDuration(Duration::days(self.lit(0, 1)?)),
                LogosToken::LitDurationW    => Token::LitDuration(Duration::weeks(self.lit(0, 1)?)),
                LogosToken::NameId          => Token::NameId(self.names.intern(self.logos.slice())),
            };
            return Ok(Some(token));
        }
        Ok(None)
//         if self.indents > 0 {
//             self.indents -= 1;
//             Ok(Some(Token::Dedent))
//         } else {
//             Ok(None)
//         }
    }
}

impl Iterator for Lexer<'_> {
    type Item = (ByteIndex, Token, ByteIndex);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.token() {
                Ok(token) => {
                    break token.map(|token| {
                        let span = self.span();
                        (span.start(), token, span.end())
                    })
                }
                Err(e) => self.diags.intern(e),
            }
        }
    }
}

/// Test for lexing Python-style semantic indentation.
#[cfg(test)] // Ignore for now
fn test() {
    let source = indoc::indoc! {"
        if true:
            if true:
                1
            else:
                2
                2
        else:
            if true:
                3
            else:
                4
    "};
    let file = 0;
    let mut names = NameInterner::default();
    let mut lexer = Lexer::new(source, 0, &mut names);

    assert!(matches!(lexer.next(), Some((_, Token::If, _))));
    assert!(matches!(lexer.next(), Some((_, Token::LitBool(true), _))));
    assert!(matches!(lexer.next(), Some((_, Token::Colon, _))));
    assert!(matches!(lexer.next(), Some((_, Token::Indent, _))));
    assert!(matches!(lexer.next(), Some((_, Token::If, _))));
    assert!(matches!(lexer.next(), Some((_, Token::LitBool(true), _))));
    assert!(matches!(lexer.next(), Some((_, Token::Colon, _))));
    assert!(matches!(lexer.next(), Some((_, Token::Indent, _))));
    assert!(matches!(lexer.next(), Some((_, Token::LitI32(1), _))));
    assert!(matches!(lexer.next(), Some((_, Token::Dedent, _))));
    assert!(matches!(lexer.next(), Some((_, Token::Else, _))));
    assert!(matches!(lexer.next(), Some((_, Token::Colon, _))));
    assert!(matches!(lexer.next(), Some((_, Token::Indent, _))));
    assert!(matches!(lexer.next(), Some((_, Token::LitI32(2), _))));
    assert!(matches!(lexer.next(), Some((_, Token::LitI32(2), _))));
    assert!(matches!(lexer.next(), Some((_, Token::Dedent, _))));
    assert!(matches!(lexer.next(), Some((_, Token::Dedent, _))));
    assert!(matches!(lexer.next(), Some((_, Token::Else, _))));
    assert!(matches!(lexer.next(), Some((_, Token::Colon, _))));
    assert!(matches!(lexer.next(), Some((_, Token::Indent, _))));
    assert!(matches!(lexer.next(), Some((_, Token::If, _))));
    assert!(matches!(lexer.next(), Some((_, Token::LitBool(true), _))));
    assert!(matches!(lexer.next(), Some((_, Token::Colon, _))));
    assert!(matches!(lexer.next(), Some((_, Token::Indent, _))));
    assert!(matches!(lexer.next(), Some((_, Token::LitI32(3), _))));
    assert!(matches!(lexer.next(), Some((_, Token::Dedent, _))));
    assert!(matches!(lexer.next(), Some((_, Token::Else, _))));
    assert!(matches!(lexer.next(), Some((_, Token::Colon, _))));
    assert!(matches!(lexer.next(), Some((_, Token::Indent, _))));
    assert!(matches!(lexer.next(), Some((_, Token::LitI32(4), _))));
    assert!(matches!(lexer.next(), Some((_, Token::Dedent, _))));
    assert!(matches!(lexer.next(), Some((_, Token::Dedent, _))));
    assert!(matches!(lexer.next(), None));
}
