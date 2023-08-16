use ast::unop;
use hir::Type;
use serde::Deserialize;
use serde::Serialize;

use ast::binop;

use super::Value;

pub fn define(builder: &mut super::Bifs) {
    builder
        .f(binop!(and), |ctx, t, v| {
            let v0 = v[0].as_bool();
            let v1 = v[1].as_bool();
            (v0 && v1).into()
        })
        .f(binop!(or), |ctx, t, v| {
            let v0 = v[0].as_bool();
            let v1 = v[1].as_bool();
            (v0 || v1).into()
        })
        .f(unop!(!), |ctx, t, v| {
            let v0 = v[0].as_bool();
            (!v0).into()
        });
}
