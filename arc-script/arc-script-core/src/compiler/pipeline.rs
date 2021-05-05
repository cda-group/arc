use crate::compiler::arcorn;
use crate::compiler::arcorn::Arcorn;
use crate::compiler::ast::AST;
use crate::compiler::hir;
use crate::compiler::hir::HIR;
use crate::compiler::info::diags::to_codespan::Report;
use crate::compiler::info::diags::Writer;
use crate::compiler::info::modes::Mode;
use crate::compiler::info::modes::Output;
use crate::compiler::info::Info;
use crate::compiler::mlir;
use crate::compiler::mlir::display::run_arc_mlir;
use crate::compiler::mlir::MLIR;

use arc_script_core_shared::Result;

use codespan_reporting::term::termcolor::WriteColor;
use tempfile::NamedTempFile;

use std::io::Write;

/// Compiles according to `opt` and writes output to `f`.
///
/// # Errors
///
/// Will return `Err` only for errors which are out of control of the compiler. In particular:
/// * Errors caused from writing to `f`.
/// * Errors caused from interactions with the filesystem.
pub fn compile<W: Writer>(mode: Mode, mut f: W) -> Result<Report> {
    //     better_panic::install();
    let mut info = Info::from(mode);

    let ast = AST::from(&mut info);

    if info.mode.fail_fast && !info.diags.is_empty() {
        if !info.mode.suppress_diags {
            info.diags.emit(&info, None, &mut f);
        }
        if info.mode.force_output {
            writeln!(f, "{}", ast.display(&info))?;
        }
        return Ok(Report::syntactic(info));
    }

    if matches!(info.mode.output, Output::AST) {
        writeln!(f, "{}", ast.display(&info))?;
        return Ok(Report::syntactic(info));
    }

    let hir = HIR::from(&ast, &mut info);

    if !info.diags.is_empty() {
        if !info.mode.suppress_diags {
            info.diags.emit(&info, &hir, &mut f);
        }
        if info.mode.force_output {
            writeln!(f, "{}", hir.display(&info))?;
        }
        return Ok(Report::semantic(info, hir));
    }

    if matches!(info.mode.output, Output::HIR) {
        writeln!(f, "{}", hir.display(&info))?;
        return Ok(Report::semantic(info, hir));
    }

    if matches!(info.mode.output, Output::Arcorn) {
        // Lower HIR into Rust
        let rust = Arcorn::from(&hir, &mut info);

        writeln!(f, "{}", rust.display())?;
        return Ok(Report::semantic(info, hir));
    }

    if matches!(info.mode.output, Output::RustMLIR) {
        // Lower HIR into Rust via MLIR
        let mlir = MLIR::from(&hir, &mut info);
        let r = mlir.display(&info);

        let infile = NamedTempFile::new().expect("Could not create temporary input file");
        let outfile = NamedTempFile::new().expect("Could not create temporary output file");
        let fw = &mut std::io::BufWriter::new(&infile);
        writeln!(fw, "{}", r)?;
        fw.flush();

        run_arc_mlir(infile.path(), outfile.path());
        let r = std::fs::read_to_string(&outfile)?;
        write!(f, "{}", r)?;
        return Ok(Report::semantic(info, hir));
    }

    if matches!(info.mode.output, Output::MLIR) {
        // Lower HIR into MLIR
        let mlir = MLIR::from(&hir, &mut info);

        writeln!(f, "{}", mlir.display(&info))?;
        return Ok(Report::semantic(info, hir));
    }

    Ok(Report::semantic(info, hir))
}
