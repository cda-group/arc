//! Conversions between `Opt` and `Mode`

use crate::repr::{self as opt, Opt, Run, SubCmd};
use arc_script_core::prelude::modes::{Input, Mode, Output};
use arc_script_core::prelude::Result;

use std::io;
use std::io::prelude::*;

impl From<Opt> for Result<Mode> {
    fn from(opt: Opt) -> Result<Mode> {
        let mut mode = match opt.subcmd {
            SubCmd::Run(cmd) => Mode {
                output: match cmd.output {
                    opt::Output::AST => Output::AST,
                    opt::Output::HIR => Output::HIR,
                    opt::Output::DFG => Output::DFG,
                    opt::Output::Rust => Output::Rust,
                    opt::Output::MLIR => Output::MLIR,
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
        };
        mode.profile = opt.profile;
        mode.verbosity = opt.verbosity;
        mode.debug = opt.debug;
        mode.fail_fast = opt.fail_fast;
        mode.suppress_diags = opt.suppress_diags;
        mode.force_output = opt.force_output;
        Ok(mode)
    }
}
