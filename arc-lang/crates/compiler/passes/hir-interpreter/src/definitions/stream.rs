use std::cell::RefCell;
use std::rc::Rc;

use hir::Name;
use hir::Type;
use im_rc::vector;
use im_rc::OrdMap;
use im_rc::Vector;
use serde::Deserialize;
use serde::Serialize;
use value::dynamic::Dataflow;
use value::dynamic::Sink;
use value::dynamic::Stream;
use value::dynamic::StreamKind;

use super::Value;

pub fn define(builder: &mut super::Bifs) {
    builder
        .f("source", |ctx, t, v| {
            let v0 = v[0].as_reader();
            let v1 = v[1].as_encoding();
            let v2 = v[2].as_time_source();
            let x = ctx.new_stream_name();
            Stream::new(x, StreamKind::DSource(v0, v1, v2).into()).into()
        })
        .f("keyby", |ctx, t, v| {
            let v0 = v[0].as_stream();
            let v1 = v[1].as_function();
            let x0 = v0.name.clone();
            let x = ctx.new_stream_name();
            v0.extend(x, StreamKind::DKeyby(x0, v1).into()).into()
        })
        .f("map", |ctx, t, v| {
            let v0 = v[0].as_stream();
            let v1 = v[1].as_function();
            let x0 = v0.name.clone();
            let x = ctx.new_stream_name();
            v0.extend(x, StreamKind::DMap(x0, v1).into()).into()
        })
        .f("filter", |ctx, t, v| {
            let v0 = v[0].as_stream();
            let v1 = v[1].as_function();
            let x0 = v0.name.clone();
            let x = ctx.new_stream_name();
            v0.extend(x, StreamKind::DFilter(x0, v1).into()).into()
        })
        .f("flatten", |ctx, t, v| {
            let v0 = v[0].as_stream();
            let x0 = v0.name.clone();
            let x = ctx.new_stream_name();
            v0.extend(x, StreamKind::DFlatten(x0).into()).into()
        })
        .f("flatmap", |ctx, t, v| {
            let v0 = v[0].as_stream();
            let v1 = v[1].as_function();
            let x0 = v0.name.clone();
            let x = ctx.new_stream_name();
            v0.extend(x, StreamKind::DFlatMap(x0, v1).into()).into()
        })
        .f("window", |ctx, t, v| {
            let v0 = v[0].as_stream();
            let v1 = v[1].as_discretizer();
            let v2 = v[2].as_aggregator();
            let x0 = v0.name.clone();
            let x = ctx.new_stream_name();
            v0.extend(x, StreamKind::DWindow(x0, v1, v2).into()).into()
        })
        .f("sink", |ctx, t, v| {
            let v0 = v[0].as_stream();
            let v1 = v[1].as_writer();
            let v2 = v[2].as_encoding();
            let x0 = v0.name.clone();
            let mut prefix = v0.prefix.clone();
            prefix.push_back(v0);
            Dataflow::new(prefix, vector![Sink::new(x0, v1, v2).into()]).into()
        });
}
