#![allow(unused)]

use std::io;
use std::io::LineWriter;
use std::io::Write;
use std::process::exit;

use anyhow::Result;
use colored::Color;
use colored::Color::Blue;
use colored::Color::Green;
use colored::Color::Red;
use colored::Colorize;
use rustyline::completion::FilenameCompleter;
use rustyline::config::Configurer;
use rustyline::error::ReadlineError;
use rustyline::highlight::MatchingBracketHighlighter;
use rustyline::hint::HistoryHinter;
use rustyline::history::FileHistory;
use rustyline::validate::MatchingBracketValidator;
use rustyline::Cmd;
use rustyline::CompletionType;
use rustyline::EditMode;
use rustyline::Editor;
use rustyline::EventHandler;

use compiler::Compiler;
use validator::StatementIterator;

use self::context::Context;

mod context;
pub mod helper;

pub fn repl(compiler: Compiler, initial: Option<String>) -> Result<()> {
    let mut ctx = Context::new(compiler)?;
    let mut stmts = initial.iter().flat_map(|s| StatementIterator::new(s));
    loop {
        let input = if let Some(stmt) = stmts.next() {
            ctx.editor.readline_with_initial(">> ", (stmt, ""))
        } else {
            ctx.editor.readline(">> ")
        };
        match input {
            Ok(input) => {
                ctx.editor.add_history_entry(&input);
                if let Err(e) = handle(&mut ctx, input) {
                    eprintln!("{}", e);
                    ctx.color(Red);
                } else {
                    ctx.color(Green);
                }
            }
            Err(ReadlineError::Interrupted) => {
                eprintln!("Interrupted");
                ctx.color(Red);
            }
            Err(ReadlineError::Eof) => break,
            Err(err) => {
                eprintln!("Error: {:?}", err);
                break;
            }
        }
    }
    Ok(())
}

fn handle(ctx: &mut Context, input: String) -> Result<()> {
    let stmts = ctx.compiler.parse(ctx.count.to_string(), input);
    let stmts = ctx.compiler.ast_to_hir(stmts);
    let stmts = ctx.compiler.infer(stmts);
    let stmts = ctx.compiler.patcomp(stmts);
    let stmts = ctx.compiler.monomorphise(stmts);
    if ctx.compiler.has_errors() {
        ctx.color(Red);
        ctx.compiler.emit_errors();
        return Ok(());
    }
    ctx.compiler.interpret(stmts);
    Ok(())
}
