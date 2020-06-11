#![warn(clippy::all)]

use crate::{ast::*, error::*, opt::*, parser::*, pretty::*, typer::*, utils::*};
use std::io::prelude::*;

#[macro_use]
mod utils;

pub mod opt;
pub mod pretty;
pub mod io;

mod ast;
mod error;
mod parser;
mod resolver;
mod typer;
mod ssa;
mod pruner;
mod mlir;

#[cfg(feature = "shaper")]
mod shaper;
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
    let (script, reporter) = compile(source, opt);

    reporter.emit();

    if !opt.debug {
        let mut w = std::io::stdout();
        match opt {
            _ if opt.mlir => writeln!(w, "{}", script.body.mlir()),
            _ => writeln!(w, "{}", script.code(opt.verbose)),
        }
        .unwrap();
        w.flush().unwrap();
    }
}

pub fn compile<'i>(source: &'i str, opt: &Opt) -> (Script, Reporter<'i>) {
    let mut parser = Parser::new();
    let mut typer = Typer::new();
    let mut script = parser.parse(source);

    if opt.debug {
        println!("=== Parsed");
        println!("{}", script.code(opt.verbose));
    }

    // script.body.download();

    script.body.assign_uid();

    if opt.debug {
        println!("=== Resolved");
        println!("{}", script.code(opt.verbose));
    }

    script.body.assign_scope();

    script.body.infer(&mut typer);

    if opt.debug {
        println!("=== Typed");
        println!("{}", script.code(opt.verbose));
    }

    script.body = script.body.into_ssa();
    script.body.prune();
    script.body.assign_scope();

    if opt.debug {
        println!("=== Typed");
        println!("{}", script.code(opt.verbose));
    }

    let errors = merge(parser.errors(), typer.errors());

    if opt.debug {
        if opt.debug {
            println!("=== Canonicalized");
            println!("{}", script.code(opt.verbose));
        }

        if errors.is_empty() {
            println!("=== MLIR");
            println!("{}", script.body.mlir());
        }
    }

    let reporter = Reporter::new(source, errors);

    (script, reporter)
}
