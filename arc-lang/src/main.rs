#![allow(unused)]

use anyhow::Result;
use clap::Parser;
use compiler::Compiler;
use repl::repl;
use std::io::Read;
use std::path::Path;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let mut logger = logging::Logger::stderr();
    let config = config::Config::try_parse()?;
    let mut compiler = Compiler::new(config, logger);

    if compiler.config.version {
        version::show();
        return Ok(());
    }

    if compiler.config.show.caches {
        compiler.show_caches();
        return Ok(());
    }
    if compiler.config.clear_caches {
        compiler.clear_caches();
        return Ok(());
    }

    compiler.compile_prelude();

    if compiler.config.file.is_some() {
        let path = compiler.config.file.as_ref().unwrap();
        let (name, source) = read_file(path)?;
        if compiler.config.interactive {
            repl(compiler, Some(source))
        } else {
            compile(compiler, name, source)
        }
    } else {
        repl(compiler, None)
    }
}

fn run(compiler: Compiler, name: String, source: String) -> Result<()> {
    if compiler.config.interactive {
        repl(compiler, Some(source))
    } else {
        compile(compiler, name, source)
    }
}

fn read_file(path: &Path) -> Result<(String, String)> {
    let name = path.display().to_string();
    let source = std::fs::read_to_string(path)?;
    if path.is_absolute() {
        std::env::set_current_dir(path.parent().unwrap())?;
    } else {
        let mut relative_path = std::env::current_dir()?;
        relative_path.push(path);
        std::env::set_current_dir(relative_path.parent().unwrap())?;
    }
    tracing::info!("Updated cwd to: {}", std::env::current_dir()?.display());
    Ok((name, source))
}

fn compile(mut compiler: Compiler, name: String, source: String) -> Result<()> {
    let ss = compiler.parse(name, source);
    let ss = compiler.ast_to_hir(ss);
    let ss = compiler.infer(ss);
    let ss = compiler.patcomp(ss);
    let ss = compiler.monomorphise(ss);
    if compiler.has_errors() {
        compiler.emit_errors();
        std::process::exit(1);
    }
    compiler.interpret(ss);
    Ok(())
}
