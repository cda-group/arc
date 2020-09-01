#![warn(clippy::all)]

use crate::{ast::*, opt::*, pretty::*};
pub use anyhow::Result;
use std::io::prelude::*;

#[macro_use]
mod utils;

#[macro_use]
extern crate educe;

pub mod opt;
pub mod pretty;
pub mod io;

mod ast;
mod error;
mod parser;
mod typer;
mod ssa;
mod pruner;
mod mlir;
mod info;
mod symbols;

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
        let mut w = std::io::stdout();
        let s = match opt {
            _ if opt.mlir => script.ast.body.mlir(&script.info),
            _ => script.code(opt.verbose, &script.info),
        };
        writeln!(w, "{}", s).unwrap();
        w.flush().unwrap();
    } else {
        script.emit_to_stdout()
    }
}

pub fn compile<'i>(source: &'i str, opt: &'i Opt) -> Script<'i> {
    let mut script = Script::parse(source);

    if opt.debug {
        println!("=== Parsed");
        println!("{}", script.code(opt.verbose, &script.info));
    }

    // script.body.download();

    script.infer();

    if opt.debug {
        println!("=== Typed");
        println!("{}", script.code(opt.verbose, &script.info));
    }

    script = script.into_ssa();
    script.prune();

    if opt.debug {
        if opt.debug {
            println!("=== Canonicalized");
            println!("{}", script.code(opt.verbose, &script.info));
        }

        if script.info.errors.is_empty() {
            println!("=== MLIR");
            println!("{}", script.ast.body.mlir(&script.info));
        }
    }

    script
}
