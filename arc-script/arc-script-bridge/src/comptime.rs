//! Macro implementation.

use arc_script_core::compiler::compile;
use arc_script_core::compiler::info::diags::sink::Buffer;
use arc_script_core::prelude::modes::{Input, Mode, Output};

use proc_macro as pm;
use proc_macro2 as pm2;
use quote::quote;

use std::fs;

/// See [`super::comptime`] for documentation.
pub(crate) fn expand(attr: pm::TokenStream, item: pm::TokenStream) -> pm::TokenStream {
    if let syn::Item::Mod(module) = syn::parse(item).expect("Expected `mod` item") {
        let (_, items) = module.content.expect("Expected `mod { ... }` found `mod;`");
        let mut args = attr.into_iter();
        let arg = args.next().expect("Expected attribute");
        args.next()
            .expect_none("Only one attribute expected, found multiple");
        let lit = syn::parse(arg.into())
            .expect("Expected a string literal attribute containing the filepath to an arc-script");
        if let syn::Lit::Str(v) = lit {
            let mut path = pm::Span::call_site().source_file().path();
            path.pop();

            let path = path.join(v.value());
            let source = fs::read_to_string(path).expect("File not found");

            let mut sink = Buffer::no_color();
            let opt = Mode {
                input: Input::Code(source),
                output: Output::Rust,
                ..Default::default()
            };

            let report = compile(opt, &mut sink).expect("Internal compiler error");
            let output = std::str::from_utf8(sink.as_slice()).expect("Internal compiler error");
            if report.is_ok() {
                let rust: syn::File = syn::parse_str(output).expect("Internal compiler error");
                let id = module.ident;
                quote!(mod #id { #(#items);* #rust }).into()
            } else {
                panic!("{}", output);
            }
        } else {
            panic!("Expected string literal")
        }
    } else {
        panic!("Expected `mod` item")
    }
}
