///! Module for constructing `AST`s in different ways.

/// Module for lexing source code into tokens.
pub mod lexer;
/// Module for parsing tokens into modules.
pub(crate) mod parser;
/// Module for importing modules and assembling a declaration-table.
pub(crate) mod importer;

use crate::compiler::ast::repr::AST;
use crate::compiler::info::modes::Input;
use crate::compiler::info::Info;

use tracing::instrument;

impl AST {
    #[instrument(name = "Info => AST", level = "debug", skip(info))]
    pub(crate) fn from(info: &mut Info) -> Self {
        tracing::debug!("{}", info);
        let mut ast = Self::default();
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
        tracing::debug!("{}", ast.debug(info));
        ast
    }
}
