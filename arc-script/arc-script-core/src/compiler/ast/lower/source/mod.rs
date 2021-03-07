///! Module for constructing `AST`s in different ways.

/// Module for lexing source code into tokens.
pub mod lexer;
/// Module for parsing tokens into modules.
pub(crate) mod parser;
/// Module for importing modules and assembling a declaration-table.
pub(crate) mod importer;
