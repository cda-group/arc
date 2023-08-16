use super::Bifs;

use super::Value;
use hir::Type;
use serde::Deserialize;
use serde::Serialize;
use std::io::Result;
use std::io::Write;

use super::*;

pub use builtins::path::Path;

pub fn define(builder: &mut super::Bifs) {
    builder.f("path", |ctx, t, v| {
        let a0 = v[0].as_string();
        Path::new(a0).into()
    });
}
