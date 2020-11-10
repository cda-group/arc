#![allow(clippy::useless_format)]
use crate::compiler::rust;
use crate::compiler::shared::display::format::Context;
use crate::compiler::shared::display::pretty::*;
use crate::compiler::shared::New;

use quote::quote;

use std::fmt::{self, Display, Formatter};
use std::fs;
use std::io;
use std::io::BufRead;
use std::io::Write;
use std::fmt::Write as FmtWrite;
use std::path::Path;
use std::process::Command;

#[derive(New, From, Copy, Clone)]
pub(crate) struct Stateless;

pub(crate) fn pretty<'i, Node>(node: &'i Node) -> Pretty<'i, Node, Stateless> {
    node.to_pretty(Stateless)
}

impl<'i> Display for Pretty<'i, rust::Rust, Stateless> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(rs, ctx) = self;
        write!(f, "{}", rustfmt(rs).unwrap())?;
        Ok(())
    }
}

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
