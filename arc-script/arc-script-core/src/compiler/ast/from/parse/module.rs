use crate::compiler::ast;
use crate::compiler::ast::from::parse::grammar::ModuleParser;
use crate::compiler::ast::from::parse::lexer::Lexer;
use crate::compiler::ast::{Item, Spanned};
use crate::compiler::info::diags::{DiagInterner, Diagnostic, Error, Result};
use crate::compiler::info::files::Loc;
use crate::compiler::info::Info;
use crate::compiler::ast::ExprInterner;

/// Parses a module
pub(crate) fn parse_module(
    name: String,
    source: String,
    exprs: &mut ExprInterner,
    info: &mut Info,
) -> Vec<Item> {
    let file_id = info.files.intern(name, source);
    let source = info.files.resolve(file_id);
    let mut lexer = Lexer::new(source, file_id, &mut info.names);
    let items = ModuleParser::new()
        .parse(file_id, exprs, &mut info.diags, &mut info.paths, &mut lexer)
        .unwrap();
    info.diags.merge(lexer.diags);
    items
}
