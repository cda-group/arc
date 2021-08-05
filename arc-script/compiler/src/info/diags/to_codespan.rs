use crate::hir;
use crate::hir::HIR;
use crate::info::diags::Diagnostic;
use crate::info::diags::Error;
use crate::info::diags::Note;
use crate::info::diags::Panic;
use crate::info::diags::Warning;
use crate::info::files::FileId;
use crate::info::files::Loc;
use crate::info::Info;

use arc_script_compiler_shared::Shrinkwrap;

use codespan_reporting::diagnostic;
use codespan_reporting::diagnostic::Label;

use std::borrow::Borrow;
use std::str;

/// A [`codespan`] diagnostic.
pub type Codespan = diagnostic::Diagnostic<FileId>;

/// A struct which contains everything necessary to print diagnostics.
/// It is intended to be used by users of the `arc_script` compiler-library.
#[derive(Debug)]
pub struct Report {
    info: Info,
    hir: Option<HIR>,
}

impl Report {
    /// Constructs a new `Report` by taking ownership of the Arc-Script compiler's output.
    /// Reports can only be constructed within the compiler, therefore this is `pub(crate)`.
    pub(crate) const fn semantic(info: Info, hir: HIR) -> Self {
        Self {
            info,
            hir: Some(hir),
        }
    }
    pub(crate) const fn syntactic(info: Info) -> Self {
        Self { info, hir: None }
    }
    /// Returns `true` if there are no diagnostics to be reported, otherwise `false`.
    pub fn is_ok(&self) -> bool {
        self.info.diags.is_empty()
    }
}

impl From<Report> for (Vec<Codespan>, Info) {
    fn from(report: Report) -> Self {
        let Report { mut info, hir } = report;
        let diags = info.diags.take();
        let ctx = &Context {
            hir: hir.as_ref(),
            info: &info,
        };
        let codepan = diags
            .store
            .into_iter()
            .filter_map(|diag| diag.to_codespan(ctx))
            .collect();
        (codepan, info)
    }
}

/// Hacky solution, HIR can optionally be omitted to
/// print diagnostics from before it was constructed.
#[derive(Debug, Copy, Clone, Shrinkwrap)]
pub struct Context<'i> {
    #[shrinkwrap(main_field)]
    pub(crate) info: &'i Info,
    pub(crate) hir: Option<&'i HIR>,
}

impl<'i> Context<'i> {
    /// Constructs a context which can be used for converting into diagnostics.
    pub(crate) const fn new(info: &'i Info, hir: Option<&'i HIR>) -> Self {
        Self { info, hir }
    }
}

/// Integration with [`codespan_reporting`].
pub trait ToCodespan {
    /// Converts `Self` into a pretty `Codespan` diagnostic.
    fn to_codespan(&self, ctx: &Context<'_>) -> Option<Codespan>;
}

impl ToCodespan for Diagnostic {
    fn to_codespan(&self, ctx: &Context<'_>) -> Option<Codespan> {
        match self {
            Diagnostic::Error(diag) => diag.to_codespan(ctx),
            Diagnostic::Warning(diag) => diag.to_codespan(ctx),
            Diagnostic::Note(diag) => diag.to_codespan(ctx),
        }
    }
}

impl ToCodespan for Note {
    fn to_codespan(&self, _ctx: &Context<'_>) -> Option<Codespan> {
        crate::todo!()
    }
}

impl ToCodespan for Warning {
    fn to_codespan(&self, _ctx: &Context<'_>) -> Option<Codespan> {
        crate::todo!()
    }
}

impl ToCodespan for Error {
    fn to_codespan(&self, ctx: &Context<'_>) -> Option<Codespan> {
        match self {
            Error::FileNotFound => Codespan::error().with_message("Source file not found."),
            Error::BadExtension => {
                Codespan::error().with_message("Input files must have `.arc` extension.")
            }
            Error::ExtraToken { found, loc } => Codespan::error()
                .with_message(format!("Extraneous token {}", found.pretty(ctx.info)))
                .with_labels(vec![label(loc)?]),
            Error::InvalidToken { loc } => Codespan::error()
                .with_message("Invalid token")
                .with_labels(vec![label(loc)?]),
            Error::UnrecognizedEOF { loc, expected } => Codespan::error()
                .with_message("Unrecognized end of file")
                .with_labels(vec![
                    label(loc)?.with_message(format!("expected {}", expected.join(", ")))
                ]),
            Error::UnrecognizedToken {
                found,
                expected,
                loc,
            } => Codespan::error()
                .with_message(format!("Unrecognized token {}", found.pretty(ctx.info)))
                .with_labels(vec![
                    label(loc)?.with_message(format!("expected {}", expected.join(", ")))
                ]),
            Error::TypeMismatch { lhs, rhs, loc } => Codespan::error()
                .with_message("Type mismatch")
                .with_labels(vec![label(loc)?.with_message(format!(
                    "{} != {}",
                    ctx.hir.unwrap().pretty(lhs, ctx.info),
                    ctx.hir.unwrap().pretty(rhs, ctx.info)
                ))]),
            Error::PathNotFound { path, loc } => Codespan::error()
                .with_message(format!(
                    "Identifier `{}` not bound to anything",
                    ctx.hir.unwrap().pretty(path, ctx.info),
                ))
                .with_labels(vec![label(loc)?.with_message("Used here")]),
            Error::NonExhaustiveMatch { loc } => Codespan::error()
                .with_message("Match is non-exhaustive")
                .with_labels(vec![label(loc)?.with_message("Missing cases")]),
            Error::NameClash { name } => {
                Codespan::error()
                    .with_message("Name clash")
                    .with_labels(vec![label(name.loc)?
                        .with_message("Name shadows previous declaration".to_owned())])
            }
            Error::FieldClash { name } => Codespan::error()
                .with_message("Found duplicate key")
                .with_labels(vec![label(name.loc)?.with_message(format!(
                    "{}",
                    ctx.hir.unwrap().pretty(&name.id, ctx.info)
                ))]),
            Error::VariantClash { name } => Codespan::error()
                .with_message("Found duplicate key")
                .with_labels(vec![label(name.loc)?.with_message(format!(
                    "{}",
                    ctx.hir.unwrap().pretty(&name.id, ctx.info)
                ))]),
            Error::VariantWrongArity { path } => Codespan::error()
                .with_message("Variant constructors expect exactly one argument.")
                .with_labels(vec![label(path.loc)?.with_message(format!(
                    "{}",
                    ctx.hir.unwrap().pretty(path, ctx.info)
                ))]),
            Error::OutOfBoundsProject { loc } => Codespan::error()
                .with_message("Out of bounds projection")
                .with_labels(vec![label(loc)?]),
            Error::FieldNotFound { loc } => Codespan::error()
                .with_message("Field not found")
                .with_labels(vec![label(loc)?]),
            Error::TypeInValuePosition { loc } => Codespan::error()
                .with_message("Expected value, found type in value-position")
                .with_labels(vec![label(loc)?]),
            Error::ValueInTypePosition { loc } => Codespan::error()
                .with_message("Expected value, found type in value-position")
                .with_labels(vec![label(loc)?]),
            Error::RefutablePattern { loc } => Codespan::error()
                .with_message("Invalid pattern")
                .with_labels(vec![label(loc)?]),
            Error::TooMuchIndent { loc } => Codespan::error()
                .with_message("Too many levels of indentation. Consider un-indenting.")
                .with_labels(vec![label(loc)?]),
            Error::BadIndent { loc } => Codespan::error()
                .with_message("Incorrectly aligned indentation.")
                .with_labels(vec![label(loc)?]),
            Error::LexicalCore { err, loc } => Codespan::error()
                .with_message(lex_err(err))
                .with_labels(vec![label(loc)?]),
            Error::Time { err, loc } => Codespan::error()
                .with_message(err.to_string())
                .with_labels(vec![label(loc)?]),
            Error::UseOfMovedValue { loc0, loc1, t } => Codespan::error()
                .with_message(format!(
                    "Use of moved value, where moved value is of non-copyable type {}",
                    ctx.hir.unwrap().pretty(t, ctx.info)
                ))
                .with_labels(vec![label(loc0)?, label(loc1)?]),
            Error::DoubleUse {
                loc0,
                loc1,
                loc2,
                t,
            } => Codespan::error()
                .with_message(format!(
                    "Double use of value of non-copyable type {}",
                    ctx.hir.unwrap().pretty(t, ctx.info)
                ))
                .with_labels(vec![label(loc0)?, label(loc1)?, label(loc2)?]),
            Error::PathIsNotVariant { loc } => Codespan::error()
                .with_message("Path is not referring to a variant")
                .with_labels(vec![label(loc)?]),
            Error::PatternInExternFun { loc } => Codespan::error()
                .with_message("Extern functions may only take parameters which are not patterns")
                .with_labels(vec![label(loc)?]),
            Error::ExpectedSelector { loc } => Codespan::error()
                .with_message("Expected selector")
                .with_labels(vec![label(loc)?]),
            Error::MultipleSelectors { loc } => Codespan::error()
                .with_message("Found multiple selectors, expected just one.")
                .with_labels(vec![label(loc)?]),
            Error::ExpectedSelectableType { loc } => Codespan::error()
                .with_message("Expected selectable type.")
                .with_labels(vec![label(loc)?]),
            Error::TypeMustBeKnownAtThisPoint { loc } => Codespan::error()
                .with_message("Type must be known at this point.")
                .with_labels(vec![label(loc)?]),
        }
        .into()
    }
}

/// Returns a textual representation of an error produced by [`lexical_core`].
#[rustfmt::skip]
fn lex_err(err: &lexical_core::Error) -> &'static str {
    use lexical_core::ErrorCode::*;
    match err.code {
        Overflow                    => "Integral overflow occurred during numeric parsing.",
        Underflow                   => "Integral underflow occurred during numeric parsing.",
        InvalidDigit                => "Invalid digit found before string termination.",
        Empty                       => "Empty byte array found.",
        EmptyMantissa               => "Empty mantissa found.",
        EmptyExponent               => "Empty exponent found.",
        EmptyInteger                => "Empty integer found.",
        EmptyFraction               => "Empty fraction found.",
        InvalidPositiveMantissaSign => "Invalid positive mantissa sign was found.",
        MissingMantissaSign         => "Mantissa sign was required, but not found.",
        InvalidExponent             => "Exponent was present but not allowed.",
        InvalidPositiveExponentSign => "Invalid positive exponent sign was found.",
        MissingExponentSign         => "Exponent sign was required, but not found.",
        ExponentWithoutFraction     => "Exponent was present without fraction component.",
        InvalidLeadingZeros         => "Integer had invalid leading zeros.",
        __Nonexhaustive             => crate::todo!(),
    }
}

fn label(loc: impl Borrow<Loc>) -> Option<Label<FileId>> {
    if let Loc::Real(file, span) = loc.borrow() {
        Some(Label::primary(*file, *span))
    } else {
        None
    }
}
