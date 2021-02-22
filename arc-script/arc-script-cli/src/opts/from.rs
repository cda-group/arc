//! Conversions from `Opt` into `Mode`

use crate::opts;
use crate::opts::Opt;
use crate::opts::Run;
use crate::opts::SubCmd;
use arc_script_core::prelude::modes::{Input, Mode, Output, Verbosity};
use arc_script_core::prelude::Result;

use std::io;
use std::io::prelude::*;

impl From<Opt> for Result<Mode> {
    fn from(opt: Opt) -> Result<Mode> {
        let mut mode = match opt.subcmd {
            SubCmd::Run(cmd) => Mode {
                output: match cmd.output {
                    opts::Output::AST => Output::AST,
                    opts::Output::HIR => Output::HIR,
                    opts::Output::DFG => Output::DFG,
                    opts::Output::Rust => Output::Rust,
                    opts::Output::MLIR => Output::MLIR,
                },
                input: Input::File(cmd.main),
                ..Default::default()
            },
            #[cfg(feature = "lsp")]
            SubCmd::Lsp => Mode {
                output: Output::Silent,
                input: Input::Empty,
                ..Default::default()
            },
            #[cfg(feature = "repl")]
            SubCmd::Repl => Mode {
                output: Output::Silent,
                input: Input::Empty,
                ..Default::default()
            },
            SubCmd::Completions(_) => unreachable!(),
        };
        mode.profile = opt.profile;
        mode.verbosity = match opt.verbosity {
            0 => Verbosity::Error,
            1 => Verbosity::Warn,
            2 => Verbosity::Info,
            3 => Verbosity::Debug,
            _ => Verbosity::Trace,
        };
        mode.debug = opt.debug;
        mode.fail_fast = opt.fail_fast;
        mode.suppress_diags = opt.suppress_diags;
        mode.force_output = opt.force_output;
        Ok(mode)
    }
}
