use crate::ast::repr::AST;
use crate::info::modes::Input;
use crate::info::Info;

use tracing::instrument;

impl AST {
    #[instrument(name = "Info => AST", level = "debug", skip(info))]
    pub(crate) fn from(info: &mut Info) -> Self {
        tracing::debug!("\n{:?}", info);
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
        tracing::debug!("\n{}", ast.debug(info));
        ast
    }
}
