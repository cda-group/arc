use hir::Type;
use serde::Deserialize;
use serde::Serialize;

use crate::Value;
use ast::unop;

pub fn define(builder: &mut super::Bifs) {
    builder
        .f(unop!(-), |ctx, t, v| {
            let v0 = v[0].as_f64();
            (-v0).into()
        })
        .f(unop!(+), |ctx, t, v| {
            let v0 = v[0].as_f64();
            v0.into()
        });
}
