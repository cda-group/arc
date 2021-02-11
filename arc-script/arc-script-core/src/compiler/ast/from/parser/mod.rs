/// Module for representing the context-free grammar of Arc-Script.
pub(crate) mod grammar {
    #[allow(clippy::all)]
    include!(concat!(
        env!("OUT_DIR"),
        "/compiler/ast/from/parser/grammar.rs"
    ));
}
/// Module for translating LALRPOP's [`ParsingError`]s to Arc-[`Diagnostics`].
pub(crate) mod error;

use crate::compiler::ast;
use crate::compiler::ast::from::lexer::Lexer;
use crate::compiler::ast::from::parser::grammar::ModuleParser;
use crate::compiler::ast::ExprInterner;
use crate::compiler::ast::{Item, Spanned};
use crate::compiler::info::diags::{DiagInterner, Diagnostic, Error, Result};
use crate::compiler::info::files::Loc;
use crate::compiler::info::Info;

impl ast::Module {
    /// Parses a module
    pub(crate) fn parse(
        name: String,
        source: String,
        exprs: &mut ExprInterner,
        info: &mut Info,
    ) -> Self {
        let file_id = info.files.intern(name, source);
        let source = info.files.resolve(file_id);
        let mut lexer = Lexer::new(source, file_id, &mut info.names);
        let items = ModuleParser::new()
            .parse(file_id, exprs, &mut info.diags, &mut info.paths, &mut lexer)
            .unwrap();
        info.diags.merge(lexer.diags);
        Self::new(items)
    }
}
