use hir::Type;
use serde::Deserialize;
use serde::Serialize;

use builtins::time::Time;

use super::Value;

pub fn define(builder: &mut super::Bifs) {
    builder
        .f("now", |ctx, t, v| Time::now().into())
        .f("from_seconds", |ctx, t, v| {
            todo!()
            // let v0 = v[0].as_i64();
            // Time::from_seconds(v0)
        })
        .f("from_nanoseconds", |ctx, t, v| {
            todo!()
            // let v0 = v[0].as_i128().0;
            // Time::from_nanoseconds(v0)
        })
        .f("seconds", |ctx, t, v| {
            let v0 = v[0].as_time();
            todo!()
        })
        .f("nanoseconds", |ctx, t, v| {
            let v0 = v[0].as_time();
            todo!()
        })
        .f("year", |ctx, t, v| {
            let v0 = v[0].as_time();
            todo!()
        })
        .f("into_string", |ctx, t, v| {
            let v0 = v[0].as_time();
            let v1 = v[1].as_string();
            todo!()
        })
        .f("from_string", |ctx, t, v| {
            let v0 = v[0].as_string();
            let v1 = v[1].as_string();
            Time::from_string(v0, v1).into()
        });
}
