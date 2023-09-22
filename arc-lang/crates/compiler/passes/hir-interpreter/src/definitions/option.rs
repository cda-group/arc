use super::Value;
use hir::Type;
use serde::Deserialize;
use serde::Serialize;

use builtins::option::Option;

pub fn define(builder: &mut super::Bifs) {
    builder
        .f("some", |ctx, t, v| {
            let v0 = v[0].clone();
            Option::some(v0).into()
        })
        .f("none", |ctx, t, v| Option::none().into())
        .f("is_some", |ctx, t, v| {
            let v0 = v[0].as_option();
            v0.is_some().into()
        })
        .f("is_none", |ctx, t, v| {
            let v0 = v[0].as_option();
            v0.is_none().into()
        })
        .f("unwrap", |ctx, t, v| {
            let v0 = v[0].as_option();
            v0.unwrap()
        });
}
