/// Module for importing modules and assembling a declaration-table.
pub(crate) mod importer;
/// Module for lexing source code into tokens.
pub mod lexer;
/// Module for representing the context-free grammar of Arc-Script.
pub(crate) mod grammar {
    #[allow(clippy::all)]
    include!(concat!(
        env!("OUT_DIR"),
        "/compiler/ast/from/parse/grammar.rs"
    ));
}
/// Module for translating LALRPOP's [`ParsingError`]s to Arc-[`Diagnostics`].
pub(crate) mod error;
/// Module for parsing tokens into an [`crate::repr::ast::AST`].
pub(crate) mod module;
/// Module for defining the format of number literals.
pub(crate) mod numfmt;
/// Module for defining tokens.
pub(crate) mod tokens;

use crate::compiler::ast::repr::AST;
use crate::compiler::info::modes::{Input, Mode};
use crate::compiler::info::Info;

impl AST {
    /// Constructs a AST from the command-line options.
    pub(super) fn parse(info: &mut Info) -> Self {
        let mut ast = Self::default();
        // TODO: Post-process the opt instead of storing it in the Info as-is.
        match &mut info.mode.input {
            Input::Code(source) => {
                let source = std::mem::take(source);
                ast.parse_source(source, info);
            }
            #[cfg(not(target_arch = "wasm32"))]
            Input::File(path) => {
                let path = std::mem::take(path);
                ast.parse_path(path, info);
            }
            Input::Empty => {}
        }
        ast
    }
}
