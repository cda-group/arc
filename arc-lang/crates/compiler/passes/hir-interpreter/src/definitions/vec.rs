use super::Value;

use crate::Tuple;
use hir::Type;
use im_rc::vector;
use serde::Deserialize;
use serde::Serialize;

pub type Vec = builtins::vec::Vec<Value>;

pub fn define(builder: &mut super::Bifs) {
    builder
        .f("new", |ctx, t, v| Vec::new().into())
        .f("push", |ctx, t, v| {
            let v0 = v[0].clone();
            let v1 = v[1].as_vec();
            v1.push(v0).into()
        })
        .f("pop", |ctx, t, v| {
            let v0 = v[0].as_vec();
            let (a, b) = v0.pop();
            let b = b.map(Into::into);
            Tuple(vector![a.into(), b.into()]).into()
        })
        .f("len", |ctx, t, v| {
            let v0 = v[0].as_vec();
            v0.len().into()
        })
        .f("get", |ctx, t, v| {
            let v0 = v[0].as_vec();
            let v1 = v[1].as_usize();
            v0.get(v1).map(Into::into).into()
        })
        .f("insert", |ctx, t, v| {
            let v0 = v[0].as_vec();
            let v1 = v[1].as_usize();
            let v2 = v[2].clone();
            v0.insert(v1, v2).into()
        })
        .f("is_empty", |ctx, t, v| {
            let v0 = v[0].as_vec();
            v0.is_empty().into()
        })
        .f("sort", |ctx, t, v| {
            let v0 = v[0].as_vec();
            todo!()
            // VVec(v0.sort())
        });
}
