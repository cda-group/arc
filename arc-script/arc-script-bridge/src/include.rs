use arc_script_core::compiler::compile;
use arc_script_core::compiler::info::diags::sink::Buffer;
use arc_script_core::prelude::modes::{Input, Mode, Output};

use proc_macro as pm;
use quote::quote;

use std::fs;
use std::path::PathBuf;

pub(crate) fn expand(attr: pm::TokenStream, item: pm::TokenStream) -> pm::TokenStream {
    if let syn::Item::Mod(item) = syn::parse::<syn::Item>(item).unwrap() {
        let ident = item.ident;
        let attrs = item.attrs;
        let vis = item.vis;
        let (_, content) = item.content.expect("Error: Expected `{}`, found `;`");
        let mod_token = item.mod_token;

        let mut file_path = PathBuf::from(pm::Span::call_site().source_file().path());
        let mut file_path = file_path.components();
        file_path.next();
        let mut file_path: PathBuf = file_path.collect();

        let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        let script_name = syn::parse::<syn::LitStr>(attr).unwrap();

        file_path.pop();
        file_path.push(script_name.value());
        file_path.set_extension("rs");

        let source_path = out_dir.join(file_path);
        let source_path_str = source_path.to_str().unwrap();

        quote!(
            #[allow(unused)]
            #vis #mod_token #ident {
                #(#content),*
                include!(#source_path_str);
            }
        )
        .into()
    } else {
        panic!("Expected module")
    }
}
