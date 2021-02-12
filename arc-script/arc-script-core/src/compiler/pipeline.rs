use crate::compiler::ast::{self, AST};
use crate::compiler::dfg::{self, DFG};
use crate::compiler::hir::{self, HIR};
use crate::compiler::info::diags::to_codespan::Report;
use crate::compiler::info::modes::{Mode, Output};
use crate::compiler::info::Info;
use crate::compiler::mlir::{self, MLIR};
use crate::compiler::rust::{self, Rust};

use codespan_reporting::term::termcolor::WriteColor;
use std::io::Write;

/// Compiles according to `opt` and writes output to `f`.
///
/// # Errors
///
/// Will return `Err` only for errors which are out of control of the compiler. In particular:
/// * Errors caused from writing to `f`.
/// * Errors caused from interactions with the filesystem.
pub fn compile<W>(mode: Mode, mut f: W) -> anyhow::Result<Report>
where
    W: Write + WriteColor,
{
    //     better_panic::install();
    tracing::debug!("{:?}", mode);

    let mut info = tracing::info_span!("Mode => Info").in_scope(|| Info::from(mode));

    let mut ast = tracing::debug_span!("Code => AST").in_scope(|| AST::from(&mut info));
    tracing::debug!("{}", info);
    tracing::debug!("{}", ast.debug(&info));

    if info.mode.fail_fast && !info.diags.is_empty() {
        if !info.mode.suppress_diags {
            info.diags.emit(&info, None, &mut f);
        }
        if info.mode.force_output {
            writeln!(f, "{}", ast.pretty(&ast, &info))?;
        }
        return Ok(Report::new(info, None));
    }

    if matches!(info.mode.output, Output::AST) {
        writeln!(f, "{}", ast.pretty(&ast, &info))?;
        return Ok(Report::new(info, None));
    }

    tracing::debug!("{}", info);

    let mut hir = tracing::debug_span!("AST => HIR").in_scope(|| HIR::from(&ast, &mut info));
    tracing::trace!("{}", hir.debug(&info));

    if !info.diags.is_empty() {
        if !info.mode.suppress_diags {
            info.diags.emit(&info, Some(&hir), &mut f);
        }
        if info.mode.force_output {
            writeln!(f, "{}", hir::pretty(&hir, &hir, &info))?;
        }
        return Ok(Report::new(info, Some(hir)));
    }

    if matches!(info.mode.output, Output::HIR) {
        writeln!(f, "{}", hir::pretty(&hir, &hir, &info))?;
        return Ok(Report::new(info, Some(hir)));
    }

    // TODO: Staging is temporarily out of order, a service squad is on its way
    //
    // Lower HIR into DFG
    //     let dfg = tracing::debug_span!("HIR => DFG").in_scope(|| {
    //         DFG::from(&hir, &mut info).unwrap_or_else(|diags| {
    //             diags.emit(&info, Some(&hir), &mut f);
    //             std::process::exit(-1);
    //         })
    //     });
    //
    //     if matches!(info.mode.output, Output::DFG) {
    //         writeln!(f, "{}", dfg::pretty(&dfg, &info))?;
    //         return Ok(Report::new(info, Some(hir)));
    //     }

    if matches!(info.mode.output, Output::Rust) {
        // Lower HIR and DFG into Rust
        let rust = tracing::debug_span!("HIR & DFG => Rust").in_scope(|| Rust::from(&hir, &info));

        writeln!(f, "{}", rust::pretty(&rust))?;
        return Ok(Report::new(info, Some(hir)));
    }

    if matches!(info.mode.output, Output::MLIR) {
        // Lower HIR and DFG into MLIR
        let mlir =
            tracing::debug_span!("HIR & DFG => MLIR").in_scope(|| MLIR::from(&hir, &mut info));

        writeln!(f, "{}", mlir::pretty(&mlir, &info))?;
        return Ok(Report::new(info, Some(hir)));
    }

    Ok(Report::new(info, Some(hir)))
}
