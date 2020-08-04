use anyhow::Result;
use arc_script::{diagnose, io::*, opt::*};

pub fn main() -> Result<()> {
    let opt = &Opt::parse();
    match &opt.subcmd {
        SubCmd::File(c) => diagnose(&read_file(&c.path)?, opt),
        SubCmd::Code(c) => diagnose(&c.code, opt),
        #[cfg(feature = "lsp")]
        SubCmd::Lsp => arc_script::lsp::lsp(opt),
        #[cfg(feature = "repl")]
        SubCmd::Repl => arc_script::repl::repl(opt)?,
        SubCmd::Lib => unreachable!(),
    }
    Ok(())
}
