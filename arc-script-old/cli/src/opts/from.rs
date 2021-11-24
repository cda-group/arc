//! Conversions from `Opt` into `Mode`

use crate::opts;
use crate::opts::Opt;
use crate::opts::Run;
use crate::opts::SubCmd;
use arc_script_compiler::prelude::modes::Input;
use arc_script_compiler::prelude::modes::Lang;
use arc_script_compiler::prelude::modes::Mode;
use arc_script_compiler::prelude::modes::Output;
use arc_script_compiler::prelude::modes::Verbosity;
use arc_script_compiler::prelude::Result;

use std::io;
use std::io::prelude::*;

impl From<Opt> for Result<Mode> {
    fn from(opt: Opt) -> Result<Mode> {
        let Opt {
            lang,
            debug,
            profile,
            fail_fast,
            suppress_diags,
            verbosity,
            force_output,
            subcmd,
            no_infer,
            no_prelude,
        } = opt;

        let mut mode = match subcmd {
            SubCmd::Run(cmd) => Mode {
                output: match cmd.output {
                    opts::Output::AST => Output::AST,
                    opts::Output::HIR => Output::HIR,
                    opts::Output::Rust => Output::Rust,
                    opts::Output::RustMLIR => Output::RustMLIR,
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
        mode.lang = match lang {
            opts::Lang::Arc => Lang::Arc,
            opts::Lang::Arq => Lang::Arq,
        };
        mode.profile = profile;
        mode.verbosity = match opt.verbosity {
            0 => Verbosity::Info,
            1 => Verbosity::Debug,
            _ => Verbosity::Trace,
        };
        mode.debug = debug;
        mode.fail_fast = fail_fast;
        mode.suppress_diags = suppress_diags;
        mode.force_output = force_output;
        mode.no_infer = no_infer;
        mode.no_prelude = no_prelude;
        Ok(mode)
    }
}
