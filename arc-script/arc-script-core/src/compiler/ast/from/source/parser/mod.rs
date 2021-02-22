#![allow(macro_use_extern_crate)]
/// Module for representing the context-free grammar of Arc-Script.
pub(crate) mod grammar {
    #![allow(unused_extern_crates)]
    #![allow(unreachable_pub)]
    #![allow(clippy::correctness)]
    #![allow(clippy::style)]
    #![allow(clippy::complexity)]
    #![allow(clippy::perf)]
    #![allow(clippy::pedantic)]
    #![allow(clippy::nursery)]
    #![allow(clippy::cargo)]
    include!(concat!(
        env!("OUT_DIR"),
        "/compiler/ast/from/source/parser/grammar.rs"
    ));
}
/// Module for translating LALRPOP's [`ParsingError`]s to Arc-[`Diagnostics`].
pub(crate) mod error;

use crate::compiler::ast;
use crate::compiler::ast::from::source::lexer::Lexer;
use crate::compiler::ast::from::source::parser::grammar::ModuleParser;
use crate::compiler::ast::ExprInterner;

use crate::compiler::info::Info;

impl ast::Module {
    /// Parses a source file with `name` that contains `text`.
    pub(crate) fn parse(
        name: String,
        text: String,
        exprs: &mut ExprInterner,
        info: &mut Info,
    ) -> Self {
        let file_id = info.files.intern(name, text);
        let source = info.files.resolve(file_id);
        let mut lexer = Lexer::new(source, file_id, &mut info.names);
        let items = ModuleParser::new()
            .parse(file_id, exprs, &mut info.diags, &mut info.paths, &mut lexer)
            .unwrap();
        info.diags.merge(lexer.diags);
        Self::new(items)
    }
}
