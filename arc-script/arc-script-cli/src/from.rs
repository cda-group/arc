use crate::repr::{Check, Opt, Parse, Run, SubCmd, Test};
use arc_script_core::prelude::modes::{Input, Mode, Output};
use arc_script_core::prelude::Result;

use std::io;
use std::io::prelude::*;

impl From<Opt> for Result<Mode> {
    fn from(opt: Opt) -> Result<Mode> {
        let mut mode = match opt.subcmd {
            SubCmd::Parse(cmd) => Mode {
                output: Output::AST,
                input: Input::File(cmd.main),
                ..Default::default()
            },
            SubCmd::Check(cmd) => Mode {
                output: Output::HIR,
                input: Input::File(cmd.main),
                ..Default::default()
            },
            SubCmd::Run(cmd) => Mode {
                output: Output::DFG,
                input: Input::File(cmd.main),
                ..Default::default()
            },
            SubCmd::Test(cmd) => Mode {
                output: Output::Rust,
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
        mode.verbosity = opt.verbosity;
        mode.debug = opt.debug;
        mode.fail_fast = opt.fail_fast;
        mode.suppress_diags = opt.suppress_diags;
        Ok(mode)
    }
}
