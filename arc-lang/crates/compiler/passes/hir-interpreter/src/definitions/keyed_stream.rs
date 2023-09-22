use hir::Name;
use hir::Type;
use serde::Deserialize;
use serde::Serialize;

use super::Value;

pub fn define(builder: &mut super::Bifs) {
    builder
        .f("keyed_source", |ctx, t, v| {
            let v0 = v[0].as_reader();
            let v1 = v[1].as_encoding();
            let v2 = v[2].as_time_source();
            todo!()
        })
        .f("keyed_keyby", |ctx, t, v| {
            let v0 = v[0].as_stream();
            let v1 = v[1].as_function();
            todo!()
        })
        .f("unkey", |ctx, t, v| {
            let v0 = v[0].as_stream();
            todo!()
        })
        .f("keyed_map", |ctx, t, v| {
            let v0 = v[0].as_stream();
            let v1 = v[1].as_function();
            todo!()
        })
        .f("keyed_filter", |ctx, t, v| {
            let v0 = v[0].as_stream();
            let v1 = v[1].as_function();
            todo!()
        })
        .f("keyed_sink", |ctx, t, v| {
            let v0 = v[0].as_stream();
            let v1 = v[1].as_writer();
            let v2 = v[2].as_encoding();
            todo!()
        })
        .f("keyed_flatmap", |ctx, t, v| {
            let v0 = v[0].as_stream();
            let v1 = v[1].as_function();
            todo!()
        })
        .f("keyed_flatten", |ctx, t, v| {
            let v0 = v[0].as_stream();
            todo!()
        })
        .f("keyed_window", |ctx, t, v| {
            let v0 = v[0].as_stream();
            let v1 = v[1].as_discretizer();
            let v2 = v[2].as_aggregator();
            todo!()
        });
}
