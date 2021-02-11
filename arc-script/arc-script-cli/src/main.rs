//! Command-line interface.

#![allow(unused)]

mod from;
mod repr;

use arc_script_core::prelude::compiler;
use arc_script_core::prelude::logger;
use arc_script_core::prelude::modes::Mode;
use arc_script_core::prelude::Result;

use crate::repr::{Opt, SubCmd};

use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};

use std::io;
use std::io::prelude::*;

use clap::Clap;

/// Command-line interface of `arc-script`
///
/// # Errors
///
/// Returns an `Err` if either the `logger` of `Mode` fails to initialize.
pub fn main() -> Result<()> {
    let mut opt = Opt::parse();

    if opt.debug {
        logger::init(opt.verbosity)?;
    }

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
        _ => {
            let sink = StandardStream::stdout(ColorChoice::Always);
            let mode: Result<Mode> = opt.into();
            compiler::compile(mode?, sink);
        }
    }

    Ok(())
}
