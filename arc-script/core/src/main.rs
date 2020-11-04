use anyhow::Result;
use arc_script::prelude::*;

pub fn main() -> Result<()> {
    let opt = &Opt::parse();
    match &opt.subcmd {
        SubCmd::File(c) => compiler::diagnose(&read_file(&c.path)?, opt),
        SubCmd::Code(c) => compiler::diagnose(&c.code, opt),
        #[cfg(feature = "lsp")]
        SubCmd::Lsp => lsp::start(opt),
        #[cfg(feature = "repl")]
        SubCmd::Repl => repl::start(opt)?,
        SubCmd::Lib => unreachable!(),
    }
    Ok(())
}
