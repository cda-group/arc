#![allow(macro_use_extern_crate)]

use context::Context;
use diagnostics::Diagnostics;
use im_rc::Vector;
use info::Info;
use lexer::Lexer;

pub(crate) mod grammar {
    #![allow(warnings)]
    include!(concat!(env!("OUT_DIR"), "/grammar.rs"));
}
pub mod context;
pub(crate) mod error;

pub fn parse_program(
    ctx: &mut Context,
    name: impl Into<String>,
    source: impl Into<String>,
) -> Vector<ast::Stmt> {
    let source = source.into();
    let id = ctx.sources.len();
    let mut lexer = Lexer::new(id, &source);
    let program = match grammar::ProgramParser::new().parse(id, &mut ctx.diagnostics, &mut lexer) {
        Ok(program) => program,
        Err(e) => {
            ctx.diagnostics
                .push_error(crate::error::parser_error(e, id));
            Vector::new()
        }
    };
    ctx.diagnostics.append(&mut lexer.diagnostics());
    ctx.sources.add(name, source);
    program
}

pub fn parse_splice(
    info: Info,
    diagnostics: &mut Diagnostics,
    source: &str,
) -> Option<ast::Splice> {
    let id = info.id().unwrap();
    let mut lexer = Lexer::new(id, source);
    let splice = match grammar::SpliceParser::new().parse(id, diagnostics, &mut lexer) {
        Ok(splice) => Some(splice),
        Err(e) => {
            diagnostics.push_error(crate::error::parser_error(e, id));
            None
        }
    };
    diagnostics.append(&mut lexer.diagnostics());
    splice
}
