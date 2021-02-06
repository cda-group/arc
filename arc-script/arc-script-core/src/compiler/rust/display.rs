#![allow(clippy::useless_format)]
use crate::compiler::rust;
use crate::compiler::shared::display::format::Context;
use crate::compiler::shared::display::pretty::*;
use crate::compiler::shared::New;

use quote::quote;

use std::fmt::Write as FmtWrite;
use std::fmt::{self, Display, Formatter};
use std::fs;
use std::io;
use std::io::BufRead;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use cfg_if::cfg_if;

#[derive(New, From, Copy, Clone)]
pub(crate) struct Stateless;

pub(crate) fn pretty<'i, Node>(node: &'i Node) -> Pretty<'i, Node, Stateless> {
    node.to_pretty(Stateless)
}

impl<'i> Display for Pretty<'i, rust::Rust, Stateless> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(rs, ctx) = self;
        cfg_if! {
            if #[cfg(not(target_arch = "wasm32"))] {
                write!(f, "{}", rustfmt(rs).unwrap())?;
            } else {
                rs.items
                    .iter()
                    .try_for_each(|item| write!(f, "{}", quote!(#item)))?;
            }
        }
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn rustfmt(rs: &rust::Rust) -> io::Result<String> {
    let tmp = tempfile::NamedTempFile::new()?;
    let fw = &mut std::io::BufWriter::new(&tmp);

    rs.items
        .iter()
        .try_for_each(|item| write!(fw, "{}", quote!(#item)))?;
    fw.flush();

    Command::new("rustfmt").arg(tmp.path()).spawn()?.wait()?;

    std::fs::read_to_string(&tmp)
}

// TODO: Use this code as soon as https://github.com/rust-lang/rust/issues/76904 is fixed.
// use rustfmt_nightly as rustfmt;
// impl<'i> Display for Pretty<'i, rust::Rust, Stateless> {
//     fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
//         let Pretty(rs, ctx) = self;
//         let sink = crate::compiler::info::diags::sink::Buffer::no_color();
//         rs.items
//             .iter()
//             .try_for_each(|item| write!(sink, "{}", quote!(#item)))
//             .unwrap();
//         let source = std::str::from_utf8(sink.as_slice()).unwrap();
//         let input = rustfmt::Input::Text(source.to_string());
//         let config = rustfmt::Config::default();
//         let setting = rustfmt::OperationSetting::default();
//
//         let formatted = rustfmt::format(input, config, setting).expect("Internal error");
//         formatted
//             .format_result()
//             .into_iter()
//             .try_for_each(|(name, result)| writeln!(f, "{}", result.formatted_text()))?;
//         Ok(())
//     }
// }
