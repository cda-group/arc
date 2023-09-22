use super::Value;
use hir::Type;
use serde::Deserialize;
use serde::Serialize;

pub type Result = builtins::result::Result<Value>;

pub fn define(builder: &mut super::Bifs) {
    builder
        .f("ok", |ctx, t, v| {
            let v0 = v[0].clone();
            Result::ok(v0).into()
        })
        .f("error", |ctx, t, v| {
            let v0 = v[0].as_string();
            Result::error(v0).into()
        })
        .f("is_ok", |ctx, t, v| {
            let v0 = v[0].as_result();
            v0.is_ok().into()
        })
        .f("is_error", |ctx, t, v| {
            let v0 = v[0].as_result();
            v0.is_error().into()
        })
        .f("unwrap_ok", |ctx, t, v| {
            let v0 = v[0].as_result();
            v0.unwrap_ok()
        })
        .f("unwrap_error", |ctx, t, v| {
            let v0 = v[0].as_result();
            v0.unwrap_error().into()
        });
}
