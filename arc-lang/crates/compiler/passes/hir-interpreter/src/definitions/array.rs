use std::io::Result;
use std::io::Write;

use ast::TypeKind::TArray;
use hir::Type;
use im_rc::vector;
use im_rc::Vector;
use serde::Deserialize;
use serde::Serialize;

use super::Value;

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize)]
pub struct Array(pub Vector<Value>);

pub fn define(builder: &mut super::Bifs) {
    builder
        .f("array_get", |ctx, t, v| {
            let v0 = v[0].as_array();
            let v1 = v[1].as_usize();
            v0.0[v1].clone()
        })
        .f("array_set", |ctx, t, v| {
            let mut v0 = v[0].as_array();
            let v1 = v[1].as_usize();
            let v2 = v[2].clone();
            v0.0[v1] = v2;
            v0.into()
        });
}
