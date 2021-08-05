//! Macro implementation.

use arc_script_compiler::compile;
use arc_script_compiler::info::diags::sink::Buffer;
use arc_script_compiler::prelude::modes::{Input, Mode, Output};

use proc_macro as pm;
use quote::quote;

use std::ffi::OsStr;
use std::fs;
use std::path::Component;
use std::path::PathBuf;

/// See [`super::include`] for documentation.
pub(crate) fn expand(attr: pm::TokenStream, item: pm::TokenStream) -> pm::TokenStream {
    if let syn::Item::Mod(item) = syn::parse::<syn::Item>(item).unwrap() {
        let ident = item.ident;
        let attrs = item.attrs;
        let vis = item.vis;
        let (_, content) = item.content.expect("Error: Expected `{}`, found `;`");
        let mod_token = item.mod_token;

        let mut file_path: PathBuf = pm::Span::call_site().source_file().path();
        let mut components = file_path.iter().peekable();

        /// Find source directory
        while let Some(c) = components.peek() {
            if c.to_str() == Some("src") {
                break;
            } else {
                components.next().unwrap();
            }
        }

        let mut file_path = PathBuf::from(
            components
                .map(|c| c.to_str().unwrap())
                .collect::<Vec<_>>()
                .join("/"),
        );
        let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

        let script_name = syn::parse::<syn::LitStr>(attr).unwrap();

        file_path.pop();
        file_path.push(script_name.value());
        file_path.set_extension("rs");

        let source_path = out_dir.join(file_path);
        let source_path_str = source_path.to_str().unwrap();

        quote!(
            #vis #mod_token #ident {
                #![allow(unused, non_snake_case)]
                #(#content),*
                include!(#source_path_str);
            }
        )
        .into()
    } else {
        panic!("Expected module")
    }
}
