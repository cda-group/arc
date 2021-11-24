//! Tokenizes a Rust string literal of source code.

use proc_macro::token_stream::IntoIter;
use proc_macro::{TokenStream, TokenTree};

use arc_script_compiler::prelude::diags::ColorChoice;
use arc_script_compiler::prelude::diags::StandardStream;
use arc_script_compiler::prelude::modes::{get_rust_backend, Input, Mode};

pub(crate) fn compile_literal(input: TokenStream) {
    let mut iter: IntoIter = input.into_iter();
    if let Some(token) = iter.next() {
        assert!(matches!(iter.next(), None));
        if let TokenTree::Literal(lit) = token {
            let source = lit.to_string();
            let mode = Mode {
                input: Input::Code(source),
                output: get_rust_backend(),
                ..Default::default()
            };
            let sink = StandardStream::stderr(ColorChoice::Never);
            arc_script_compiler::compile(mode, sink).unwrap();
        } else {
            panic!("Expected literal");
        }
    } else {
        panic!("Expected token");
    }
}
