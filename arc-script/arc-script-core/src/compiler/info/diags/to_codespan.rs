use crate::compiler::ast;
use crate::compiler::hir;
use crate::compiler::info::diags::{Diagnostic, Error, Note, Warning};
use crate::compiler::info::files::{FileId, Loc};
use crate::compiler::info::paths::PathId;
use crate::compiler::info::types::TypeId;
use crate::compiler::info::Info;
use crate::compiler::shared::display::pretty::AsPretty;

use codespan_reporting::diagnostic::{self, Label};
use codespan_reporting::files;
use codespan_reporting::term::termcolor::{Buffer, ColorChoice, StandardStream, WriteColor};
use codespan_reporting::term::{self, Config};
use std::borrow::Borrow;
use std::io;
use std::str;

type Codespan = diagnostic::Diagnostic<FileId>;

/// Converterts `Self` into a pretty `Codespan` diagnostic.
pub trait ToCodespan {
    fn to_codespan(&self, info: &Info) -> Option<Codespan>;
}

impl ToCodespan for Diagnostic {
    fn to_codespan(&self, info: &Info) -> Option<Codespan> {
        match self {
            Diagnostic::Error(diag) => diag.to_codespan(info),
            Diagnostic::Warning(diag) => diag.to_codespan(info),
            Diagnostic::Note(diag) => diag.to_codespan(info),
        }
    }
}

impl ToCodespan for Note {
    fn to_codespan(&self, info: &Info) -> Option<Codespan> {
        todo!()
    }
}

impl ToCodespan for Warning {
    fn to_codespan(&self, info: &Info) -> Option<Codespan> {
        todo!()
    }
}

impl ToCodespan for Error {
    fn to_codespan(&self, info: &Info) -> Option<Codespan> {
        match self {
            Error::FileNotFound => Codespan::error().with_message("Source file not found."),
            Error::BadLiteral { msg, loc } => Codespan::error()
                .with_message("Bad literal.")
                .with_labels(vec![label(loc)?.with_message(msg)]),
            Error::ExtraToken { found, loc } => Codespan::error()
                .with_message(format!("Extraneous token {}", found))
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
                .with_message(format!("Unrecognized token {}", found))
                .with_labels(vec![
                    label(loc)?.with_message(format!("expected {}", expected.join(", ")))
                ]),
            Error::TypeMismatch { lhs, rhs, loc } => Codespan::error()
                .with_message("Type mismatch")
                .with_labels(vec![label(loc)?.with_message(format!(
                    "{} != {}",
                    hir::pretty(lhs, info),
                    hir::pretty(rhs, info)
                ))]),
            Error::PathNotFound { path, loc } => Codespan::error()
                .with_message(format!(
                    "Identifier `{}` not bound to anything",
                    hir::pretty(path, info),
                ))
                .with_labels(vec![label(loc)?.with_message("Used here")]),
            Error::DisallowedDimExpr { loc } => Codespan::error()
                .with_message("Disallowed expression in dimension")
                .with_labels(vec![label(loc)?.with_message("Found here")]),
            Error::ShapeUnsat => Codespan::error().with_message("Unsatisfiable shape"),
            Error::ShapeUnknown => Codespan::error().with_message("Unknown shape"),
            Error::NonExhaustiveMatch { loc } => Codespan::error()
                .with_message("Match is non-exhaustive")
                .with_labels(vec![label(loc)?.with_message("Missing cases")]),
            Error::NameClash { name } => {
                Codespan::error()
                    .with_message("Name clash")
                    .with_labels(vec![
                        label(name.loc)?.with_message(format!("Name shadows previous declaration"))
                    ])
            }
            Error::FieldClash { name } => Codespan::error()
                .with_message("Found duplicate key")
                .with_labels(vec![
                    label(name.loc)?.with_message(format!("{}", hir::pretty(&name.id, info)))
                ]),
            Error::VariantClash { name } => Codespan::error()
                .with_message("Found duplicate key")
                .with_labels(vec![
                    label(name.loc)?.with_message(format!("{}", hir::pretty(&name.id, info)))
                ]),
            Error::OutOfBoundsProject { loc } => Codespan::error()
                .with_message("Out of bounds projection")
                .with_labels(vec![label(loc)?]),
            Error::FieldNotFound { loc } => Codespan::error()
                .with_message("Field not found")
                .with_labels(vec![label(loc)?]),
            Error::MainNotFound => Codespan::error().with_message("`main` function not found"),
            Error::MainWrongSign => Codespan::error()
                .with_message("`main` function has wrong signature, expected main() -> ()"),
            Error::CycleDetected => {
                Codespan::error().with_message("Cycle detected in the Dataflow Graph")
            }
            Error::TypeInValuePosition { loc } => Codespan::error()
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
            Error::UseOfMovedValue { loc0, loc1 } => Codespan::error()
                .with_message("Use of moved value")
                .with_labels(vec![label(loc0)?, label(loc1)?]),
            Error::DoubleUse { loc0, loc1, loc2 } => Codespan::error()
                .with_message("Double use of value")
                .with_labels(vec![label(loc0)?, label(loc1)?, label(loc2)?]),
            Error::Panic { loc, trace } => {
                let mut labels = vec![label(loc)?];
                labels.extend(trace.into_iter().enumerate().filter_map(|(i, path)| {
                    let name = info.resolve_to_names(path.id);
                    label(path.loc).map(|l| l.with_message(format!("message")))
                }));
                Codespan::error()
                    .with_message("Runtime error")
                    .with_labels(labels)
            }
            Error::PathIsNotVariant { loc } => Codespan::error()
                .with_message("Path is not referring to a variant")
                .with_labels(vec![label(loc)?]),
        }
        .into()
    }
}

#[rustfmt::skip]
fn lex_err(err: &lexical_core::Error) -> String {
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
        _ => unreachable!()
    }
    .to_owned()
}

fn label(loc: impl Borrow<Option<Loc>>) -> Option<Label<FileId>> {
    loc.borrow().map(|loc| Label::primary(loc.file, loc.span))
}
