use arc_script::{diagnose, io::*, opt::*};

pub type Error = Box<dyn std::error::Error + 'static>;

pub fn main() -> Result<(), Error> {
    let opt = &Opt::get();
    match &opt.subcmd {
        SubCmd::Repl => run_repl(opt)?,
        SubCmd::File(c) => diagnose(&read_file(&c.path)?, opt),
        SubCmd::Code(c) => diagnose(&c.code, opt),
        SubCmd::Lsp => run_lsp(opt)?,
        SubCmd::Bench => {}
        SubCmd::Lib => {}
    }
    Ok(())
}

#[cfg(feature = "lsp")]
fn run_lsp(opt: &Opt) -> Result<(), Error> {
    Ok(arc_script::lsp::lsp(opt))
}

#[cfg(not(feature = "lsp"))]
fn run_lsp(_: &Opt) -> Result<(), Error> {
    panic!("Binary was not compiled with LSP support.")
}

#[cfg(feature = "repl")]
fn run_repl(opt: &Opt) -> Result<(), Error> {
    arc_script::repl::repl(opt)
}

#[cfg(not(feature = "repl"))]
fn run_repl(_: &Opt) -> Result<(), Error> {
    panic!("Binary was not compiled with REPL support.")
}
