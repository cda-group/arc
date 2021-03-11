use crate::compiler::arcon;
use crate::compiler::arcon::Arcon;
use crate::compiler::ast::AST;
use crate::compiler::dfg;
use crate::compiler::dfg::DFG;
use crate::compiler::hir;
use crate::compiler::hir::HIR;
use crate::compiler::info::diags::to_codespan::Report;
use crate::compiler::info::modes::Mode;
use crate::compiler::info::modes::Output;
use crate::compiler::info::Info;
use crate::compiler::mlir;
use crate::compiler::mlir::display::run_arc_mlir;
use crate::compiler::mlir::MLIR;
use arc_script_core_shared::Result;

use codespan_reporting::term::termcolor::WriteColor;
use std::io::Write;

/// Compiles according to `opt` and writes output to `f`.
///
/// # Errors
///
/// Will return `Err` only for errors which are out of control of the compiler. In particular:
/// * Errors caused from writing to `f`.
/// * Errors caused from interactions with the filesystem.
pub fn compile<W>(mode: Mode, mut f: W) -> Result<Report>
where
    W: Write + WriteColor,
{
    //     better_panic::install();
    let mut info = Info::from(mode);

    let ast = AST::from(&mut info);

    if info.mode.fail_fast && !info.diags.is_empty() {
        if !info.mode.suppress_diags {
            info.diags.emit(&info, None, &mut f);
        }
        if info.mode.force_output {
            writeln!(f, "{}", ast.pretty(&ast, &info))?;
        }
        return Ok(Report::syntactic(info));
    }

    if matches!(info.mode.output, Output::AST) {
        writeln!(f, "{}", ast.pretty(&ast, &info))?;
        return Ok(Report::syntactic(info));
    }

    let hir = HIR::from(&ast, &mut info);

    if !info.diags.is_empty() {
        if !info.mode.suppress_diags {
            info.diags.emit(&info, &hir, &mut f);
        }
        if info.mode.force_output {
            writeln!(f, "{}", hir::pretty(&hir, &hir, &info))?;
        }
        return Ok(Report::semantic(info, hir));
    }

    if matches!(info.mode.output, Output::HIR) {
        writeln!(f, "{}", hir::pretty(&hir, &hir, &info))?;
        return Ok(Report::semantic(info, hir));
    }

    // TODO: Staging is temporarily out of order, a service squad is on its way
    //
    // Lower HIR into DFG
    //     let dfg = DFG::from(&hir, &mut info).unwrap_or_else(|diags| {
    //         diags.emit(&info, &hir, &mut f);
    //         std::process::exit(-1);
    //     });
    //
    // if matches!(info.mode.output, Output::DFG) {
    //     writeln!(f, "{}", dfg::pretty(&dfg, &info))?;
    //     return Ok(Report::new(info, hir));
    // }

    if matches!(info.mode.output, Output::Rust) {
        // Lower HIR and DFG into Rust
        let rust = Arcon::from(&hir, &mut info);

        writeln!(f, "{}", arcon::pretty(&rust))?;
        return Ok(Report::semantic(info, hir));
    }

    if matches!(info.mode.output, Output::RustMLIR) {
        // Lower HIR and DFG into Rust via MLIR
        let mlir = MLIR::from(&hir, &mut info);
        let r = mlir::pretty(&mlir, &info);

        let infile = tempfile::NamedTempFile::new().expect("Could not create temporary input file");
        let outfile =
            tempfile::NamedTempFile::new().expect("Could not create temporary output file");
        let fw = &mut std::io::BufWriter::new(&infile);
        writeln!(fw, "{}", r)?;
        fw.flush();

        run_arc_mlir(infile.path(), outfile.path());
	let r = std::fs::read_to_string(&outfile)?;
        write!(f, "{}", r)?;
        return Ok(Report::semantic(info, hir));
    }

    if matches!(info.mode.output, Output::MLIR) {
        // Lower HIR and DFG into MLIR
        let mlir = MLIR::from(&hir, &mut info);

        writeln!(f, "{}", mlir::pretty(&mlir, &info))?;
        return Ok(Report::semantic(info, hir));
    }

    Ok(Report::semantic(info, hir))
}
