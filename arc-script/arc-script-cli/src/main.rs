//! Command-line interface.

#![allow(unused)]

/// Module for logging debug information.
pub mod logger;
/// Module for representing CLI options.
mod repr;
/// Module for converting CLI options to compiler `Mode`s.
mod from;

use crate::repr::Opt;
use crate::repr::SubCmd;
use arc_script_core::prelude::compiler;
use arc_script_core::prelude::modes::Mode;
use arc_script_core::prelude::Result;

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
    run().and(Ok(()))
}

/// Wrapper on top of main which ensures that `guard` is dropped when the
/// function returns.
fn run() -> Result<Option<impl Drop>> {
    let mut opt = Opt::parse();

    let guard = if opt.debug {
        Some(logger::init(opt.verbosity)?)
    } else {
        None
    };

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

    Ok(guard)
}
