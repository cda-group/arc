//! Macro implementation.

use arc_script_core::compiler::compile;
use arc_script_core::compiler::info::diags::sink::Buffer;
use arc_script_core::prelude::modes::{get_rust_backend, Input, Mode, Output};

use proc_macro as pm;
use proc_macro2 as pm2;
use quote::quote;

use std::fs;

/// See [`super::compile`] for documentation.
pub(crate) fn expand(attr: pm::TokenStream, item: pm::TokenStream) -> pm::TokenStream {
    if let syn::Item::Mod(item) = syn::parse(item).expect("Expected `mod` item") {
        let ident = item.ident;
        let attrs = item.attrs;
        let vis = item.vis;
        let (_, content) = item.content.expect("Error: Expected `{}`, found `;`");
        let mod_token = item.mod_token;

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
                output: get_rust_backend(),
                ..Default::default()
            };

            let report = compile(opt, &mut sink).expect("Internal compiler error");
            let output = std::str::from_utf8(sink.as_slice()).expect("Internal compiler error");
            if report.is_ok() {
                let rust: syn::File = syn::parse_str(output).expect("Internal compiler error");

                quote!(
                    #vis #mod_token #ident {
                        #![allow(unused, non_snake_case)]
                        #(#content),*
                        #rust
                    }
                )
                .into()
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
