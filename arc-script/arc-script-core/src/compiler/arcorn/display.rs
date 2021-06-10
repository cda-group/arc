#![allow(clippy::useless_format)]

#[path = "../pretty.rs"]
pub(crate) mod pretty;

use pretty::*;

use crate::compiler::arcorn;

use arc_script_core_shared::cfg_if;
use arc_script_core_shared::From;
use arc_script_core_shared::New;

use quote::quote;

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Write as FmtWrite;
use std::io;
use std::io::Write;
use std::process::Command;

#[derive(New, From, Copy, Clone)]
pub(crate) struct Context;

impl arcorn::Arcorn {
    pub(crate) fn display(&self) -> Pretty<'_, Self, Context> {
        self.to_pretty(Context)
    }
}

impl<'i> Display for Pretty<'i, arcorn::Arcorn, Context> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let rs = self.node;
        let fmt = self.fmt;
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
fn rustfmt(rs: &arcorn::Arcorn) -> io::Result<String> {
    let tmp = tempfile::NamedTempFile::new()?;
    let fw = &mut std::io::BufWriter::new(&tmp);

    let file = &rs.file;

    write!(fw, "{}", quote::quote!(#file))?;
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
