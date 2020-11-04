#![deny(clippy::all)]

use crate::prelude::*;
pub use anyhow::Result;

#[macro_use]
mod utils;

#[macro_use]
extern crate educe;

pub mod codegen;
pub mod io;
pub mod opt;

mod ast;
mod connector;
mod dataflow;
mod error;
mod eval;
mod info;
mod lexer;
mod parser;
mod prelude;
mod pruner;
mod ssa;
mod symbols;
mod typer;

// #[cfg(feature = "shaper")]
// mod shaper;
#[cfg(feature = "lsp")]
mod completer;
#[cfg(feature = "lsp")]
mod linter;
#[cfg(feature = "lsp")]
pub mod lsp;
#[cfg(feature = "manifest")]
mod manifest;
#[cfg(feature = "provider")]
mod provider;
#[cfg(feature = "repl")]
pub mod repl;

pub fn diagnose(source: &str, opt: &Opt) {
    let script = compile(source, opt);
    if script.info.errors.is_empty() && !opt.debug {
        if opt.mlir {
            println!("{}", script.mlir());
        } else {
            println!("{}", script);
        };
    } else {
        script.emit_to_stdout()
    }
}

pub fn compile<'i>(source: &'i str, opt: &'i Opt) -> Script<'i> {
    let mut script = Script::parse(source, opt);

    if opt.debug {
        println!("=== Opt");
        println!("{:?}", opt);
        println!("=== Parsed");
        println!("{}", script);
    }

    // script.body.download();

    script.infer();

    if opt.debug {
        println!("=== Typed");
        println!("{}", script);
    }

    script = script.into_ssa();
    script.prune();

    if opt.debug {
        if opt.debug {
            println!("=== Canonicalized");
            println!("{}", script);
        }

        if script.info.errors.is_empty() {
            println!("=== MLIR");
            println!("{}", script.mlir());
        }
    }

    if !opt.check {
        let dataflow = script.eval();
        println!("{}", dataflow.pretty());
    }

    script
}
