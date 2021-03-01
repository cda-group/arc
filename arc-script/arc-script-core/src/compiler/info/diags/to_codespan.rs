use crate::compiler::hir;
use crate::compiler::hir::HIR;
use crate::compiler::info::diags::Diagnostic;
use crate::compiler::info::diags::Error;
use crate::compiler::info::diags::Note;
use crate::compiler::info::diags::Panic;
use crate::compiler::info::diags::Warning;
use crate::compiler::info::files::FileId;
use crate::compiler::info::files::Loc;

use crate::compiler::info::Info;
use crate::compiler::pretty::AsPretty;

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
#[derive(Debug, Copy, Clone)]
pub struct Context<'i> {
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
            Self::Error(diag) => diag.to_codespan(ctx),
            Self::Warning(diag) => diag.to_codespan(ctx),
            Self::Note(diag) => diag.to_codespan(ctx),
            Self::Panic(diag) => diag.to_codespan(ctx),
        }
    }
}

impl ToCodespan for Note {
    fn to_codespan(&self, _ctx: &Context<'_>) -> Option<Codespan> {
        todo!()
    }
}

impl ToCodespan for Warning {
    fn to_codespan(&self, _ctx: &Context<'_>) -> Option<Codespan> {
        todo!()
    }
}

impl ToCodespan for Error {
    fn to_codespan(&self, ctx: &Context<'_>) -> Option<Codespan> {
        match self {
            Self::FileNotFound => Codespan::error().with_message("Source file not found."),
            Self::ExtraToken { found, loc } => Codespan::error()
                .with_message(format!("Extraneous token {}", found.pretty(ctx.info)))
                .with_labels(vec![label(loc)?]),
            Self::InvalidToken { loc } => Codespan::error()
                .with_message("Invalid token")
                .with_labels(vec![label(loc)?]),
            Self::UnrecognizedEOF { loc, expected } => Codespan::error()
                .with_message("Unrecognized end of file")
                .with_labels(vec![
                    label(loc)?.with_message(format!("expected {}", expected.join(", ")))
                ]),
            Self::UnrecognizedToken {
                found,
                expected,
                loc,
            } => Codespan::error()
                .with_message(format!("Unrecognized token {}", found.pretty(ctx.info)))
                .with_labels(vec![
                    label(loc)?.with_message(format!("expected {}", expected.join(", ")))
                ]),
            Self::TypeMismatch { lhs, rhs, loc } => Codespan::error()
                .with_message("Type mismatch")
                .with_labels(vec![label(loc)?.with_message(format!(
                    "{} != {}",
                    hir::pretty(lhs, ctx.hir.unwrap(), ctx.info),
                    hir::pretty(rhs, ctx.hir.unwrap(), ctx.info)
                ))]),
            Self::PathNotFound { path, loc } => Codespan::error()
                .with_message(format!(
                    "Identifier `{}` not bound to anything",
                    hir::pretty(path, ctx.hir.unwrap(), ctx.info),
                ))
                .with_labels(vec![label(loc)?.with_message("Used here")]),
            Self::DisallowedDimExpr { loc } => Codespan::error()
                .with_message("Disallowed expression in dimension")
                .with_labels(vec![label(loc)?.with_message("Found here")]),
            Self::ShapeUnsat => Codespan::error().with_message("Unsatisfiable shape"),
            Self::ShapeUnknown => Codespan::error().with_message("Unknown shape"),
            Self::NonExhaustiveMatch { loc } => Codespan::error()
                .with_message("Match is non-exhaustive")
                .with_labels(vec![label(loc)?.with_message("Missing cases")]),
            Self::NameClash { name } => {
                Codespan::error()
                    .with_message("Name clash")
                    .with_labels(vec![label(name.loc)?
                        .with_message("Name shadows previous declaration".to_owned())])
            }
            Self::FieldClash { name } => Codespan::error()
                .with_message("Found duplicate key")
                .with_labels(vec![label(name.loc)?.with_message(format!(
                    "{}",
                    hir::pretty(&name.id, ctx.hir.unwrap(), ctx.info)
                ))]),
            Self::VariantClash { name } => Codespan::error()
                .with_message("Found duplicate key")
                .with_labels(vec![label(name.loc)?.with_message(format!(
                    "{}",
                    hir::pretty(&name.id, ctx.hir.unwrap(), ctx.info)
                ))]),
            Self::VariantWrongArity { path } => Codespan::error()
                .with_message("Variant constructors expect exactly one argument.")
                .with_labels(vec![label(path.loc)?.with_message(format!(
                    "{}",
                    hir::pretty(&path.id, ctx.hir.unwrap(), ctx.info)
                ))]),
            Self::OutOfBoundsProject { loc } => Codespan::error()
                .with_message("Out of bounds projection")
                .with_labels(vec![label(loc)?]),
            Self::FieldNotFound { loc } => Codespan::error()
                .with_message("Field not found")
                .with_labels(vec![label(loc)?]),
            Self::MainNotFound => Codespan::error().with_message("`main` function not found"),
            Self::MainWrongSign => Codespan::error()
                .with_message("`main` function has wrong signature, expected main() -> ()"),
            Self::CycleDetected => {
                Codespan::error().with_message("Cycle detected in the Dataflow Graph")
            }
            Self::TypeInValuePosition { loc } => Codespan::error()
                .with_message("Expected value, found type in value-position")
                .with_labels(vec![label(loc)?]),
            Self::RefutablePattern { loc } => Codespan::error()
                .with_message("Invalid pattern")
                .with_labels(vec![label(loc)?]),
            Self::TooMuchIndent { loc } => Codespan::error()
                .with_message("Too many levels of indentation. Consider un-indenting.")
                .with_labels(vec![label(loc)?]),
            Self::BadIndent { loc } => Codespan::error()
                .with_message("Incorrectly aligned indentation.")
                .with_labels(vec![label(loc)?]),
            Self::LexicalCore { err, loc } => Codespan::error()
                .with_message(lex_err(err))
                .with_labels(vec![label(loc)?]),
            Self::UseOfMovedValue { loc0, loc1, tv } => Codespan::error()
                .with_message(format!(
                    "Use of moved value, where moved value is of non-copyable type {}",
                    hir::pretty(tv, ctx.hir.unwrap(), ctx.info)
                ))
                .with_labels(vec![label(loc0)?, label(loc1)?]),
            Self::DoubleUse {
                loc0,
                loc1,
                loc2,
                tv,
            } => Codespan::error()
                .with_message(format!(
                    "Double use of value of non-copyable type {}",
                    hir::pretty(tv, ctx.hir.unwrap(), ctx.info)
                ))
                .with_labels(vec![label(loc0)?, label(loc1)?, label(loc2)?]),
            Self::PathIsNotVariant { loc } => Codespan::error()
                .with_message("Path is not referring to a variant")
                .with_labels(vec![label(loc)?]),
        }
        .into()
    }
}

impl ToCodespan for Panic {
    fn to_codespan(&self, ctx: &Context<'_>) -> Option<Codespan> {
        match self {
            Self::Unwind { loc, trace } => {
                let mut labels = vec![label(loc)?];
                labels.extend(trace.iter().enumerate().filter_map(|(_i, path)| {
                    let _name = ctx.info.resolve_to_names(path.id);
                    label(path.loc).map(|l| l.with_message("Runtime error thrown here"))
                }));
                Codespan::error()
                    .with_message("Runtime error")
                    .with_labels(labels)
            }
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
        __Nonexhaustive             => todo!(),
    }
}

fn label(loc: impl Borrow<Option<Loc>>) -> Option<Label<FileId>> {
    loc.borrow().map(|loc| Label::primary(loc.file, loc.span))
}
