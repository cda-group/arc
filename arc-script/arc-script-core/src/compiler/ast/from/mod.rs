mod parse;

use crate::compiler::ast::repr::AST;
use crate::compiler::info::Info;

impl From<&'_ mut Info> for AST {
    /// Constructs a AST from the command-line options.
    fn from(info: &mut Info) -> Self {
        let ast = AST::parse(info);
        ast
    }
}
