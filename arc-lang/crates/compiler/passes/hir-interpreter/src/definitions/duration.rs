use hir::Type;
use serde::Deserialize;
use serde::Serialize;
use std::io::Result;
use std::io::Write;

use builtins::duration::Duration;

use super::Value;

pub fn define(builder: &mut super::Bifs) {
    builder
        .f("from_seconds", |ctx, t, v| {
            let v0 = v[0].as_i64();
            Duration::from_seconds(v0).into()
        })
        .f("from_milliseconds", |ctx, t, v| {
            let v0 = v[0].as_i64();
            Duration::from_milliseconds(v0).into()
        })
        .f("from_microseconds", |ctx, t, v| {
            let v0 = v[0].as_i64();
            Duration::from_microseconds(v0).into()
        })
        .f("from_nanoseconds", |ctx, t, v| {
            let v0 = v[0].as_i64();
            Duration::from_nanoseconds(v0).into()
        })
        .f("__s", |ctx, t, v| {
            let v0 = v[0].as_i32();
            Duration::from_seconds(v0 as i64).into()
        })
        .f("__ms", |ctx, t, v| {
            let v0 = v[0].as_i32();
            Duration::from_seconds(v0 as i64).into()
        })
        .f("__us", |ctx, t, v| {
            let v0 = v[0].as_i32();
            Duration::from_seconds(v0 as i64).into()
        })
        .f("__ns", |ctx, t, v| {
            let v0 = v[0].as_i32();
            Duration::from_seconds(v0 as i64).into()
        });
}
