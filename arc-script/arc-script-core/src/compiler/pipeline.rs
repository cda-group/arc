use crate::compiler::ast::{self, AST};
use crate::compiler::dfg::{self, DFG};
use crate::compiler::hir::{self, HIR};
use crate::compiler::info::modes::{Mode, Output};
use crate::compiler::info::Info;
use crate::compiler::mlir::{self, MLIR};
use crate::compiler::rust::{self, Rust};

use codespan_reporting::term::termcolor::WriteColor;
use std::io::Write;

/// Compiles according to `opt` and writes output to `f`.
pub fn compile<W>(mode: Mode, mut f: W) -> anyhow::Result<Info>
where
    W: Write + WriteColor,
{
    tracing::debug!("{:#?}", mode);
    tracing::debug!("Parsing AST...");

    let mut info = Info::from(mode);
    let mut ast = AST::from(&mut info);
    tracing::trace!("{:#?}", ast);

    if info.mode.fail_fast && !info.diags.is_empty() {
        if !info.mode.suppress_diags {
            info.diags.emit_to_stdout(&info, &mut f);
        }
        if info.mode.force_output {
            writeln!(f, "{}", ast.pretty(&ast, &mut info))?;
        }
        return Ok(info);
    }

    if matches!(info.mode.output, Output::AST) {
        writeln!(f, "{}", ast.pretty(&ast, &mut info))?;
        return Ok(info);
    }

    tracing::debug!("{:?}", info);
    tracing::debug!("Lowering AST to HIR...");

    let mut hir = HIR::from(&ast, &mut info);
    tracing::trace!("{:#?}", hir);

    if !info.diags.is_empty() {
        if !info.mode.suppress_diags {
            info.diags.emit_to_stdout(&info, &mut f);
        }
        if info.mode.force_output {
            writeln!(f, "{}", ast.pretty(&ast, &mut info))?;
        }
        return Ok(info);
    }

    if matches!(info.mode.output, Output::HIR) {
        writeln!(f, "{}", hir::pretty(&hir, &mut info))?;
        return Ok(info);
    }

    tracing::debug!("Lowering HIR to DFG...");
    // Lower HIR into DFG
    let dfg = DFG::from(&hir, &mut info).unwrap_or_else(|diags| {
        diags.emit_to_stdout(&info, &mut f);
        std::process::exit(-1);
    });

    if matches!(info.mode.output, Output::DFG) {
        writeln!(f, "{}", dfg::pretty(&dfg, &info))?;
        return Ok(info);
    }

    if matches!(info.mode.output, Output::Rust) {
        tracing::debug!("Lowering HIR and DFG into Rust...");
        // Lower HIR and DFG into Rust
        let rust = Rust::from(&hir, &dfg, &info);

        writeln!(f, "{}", rust::pretty(&rust))?;
        return Ok(info);
    }

    if matches!(info.mode.output, Output::MLIR) {
        tracing::debug!("Lowering HIR and DFG into MLIR...");
        // Lower HIR and DFG into MLIR
        let mlir = MLIR::from(hir, dfg, &mut info);

        writeln!(f, "{}", mlir::pretty(&mlir, &mut info))?;
        return Ok(info);
    }

    Ok(info)
}
