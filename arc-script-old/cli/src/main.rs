//! Command-line interface.

#![allow(unused)]

/// Module for logging debug information.
pub mod logger;
mod opts;

use crate::opts::Opt;
use crate::opts::Shell;
use crate::opts::SubCmd;
use arc_script_compiler::prelude;
use arc_script_compiler::prelude::modes::Mode;
use arc_script_compiler::prelude::Result;

use clap::App;
use clap::Clap;
use clap::IntoApp;
use clap_generate::generate;
use clap_generate::generators::Bash;
use clap_generate::generators::Elvish;
use clap_generate::generators::Fish;
use clap_generate::generators::PowerShell;
use clap_generate::generators::Zsh;
use codespan_reporting::term::termcolor::ColorChoice;
use codespan_reporting::term::termcolor::StandardStream;

use std::io;
use std::io::prelude::*;

/// Command-line interface of `arc-script`
///
/// # Errors
///
/// Returns an `Err` if either the `logger` of `Mode` fails to initialize.
pub fn main() -> Result<()> {
    run().and(Ok(()))
}

/// Wrapper on top of main which ensures that `guard` is dropped when the
/// function returns.
fn run() -> Result<Option<impl Drop>> {
    let mut opt = Opt::parse();

    let guard = logger::init(&opt)?;

    match opt.subcmd {
        #[cfg(feature = "lsp")]
        SubCmd::Lsp => {
            let mode: Result<Mode> = opt.into();
            arc_script_lsp::start(mode?);
        }
        #[cfg(feature = "repl")]
        SubCmd::Repl => {
            let mode: Result<Mode> = opt.into();
            arc_script_repl::start(mode?)?;
        }
        SubCmd::Completions(subcmd) => subcmd.generate(),
        SubCmd::Run(_) => {
            let sink = StandardStream::stdout(ColorChoice::Always);
            let mode: Result<Mode> = opt.into();
            arc_script_compiler::compile(mode?, sink);
        }
    }

    Ok(guard)
}
