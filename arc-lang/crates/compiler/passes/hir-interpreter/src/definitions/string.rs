use hir::Type;
use im_rc::vector;
use serde::Deserialize;
use serde::Serialize;

use crate::Tuple;
use builtins::string::String;
use builtins::vec::Vec;

use super::Value;

pub fn define(builder: &mut super::Bifs) {
    builder
        .f("new", |ctx, t, v| String::new().into())
        .f("with_capacity", |ctx, t, v| {
            let a0 = v[0].as_usize();
            String::with_capacity(a0).into()
        })
        .f("push_char", |ctx, t, v| {
            let a0 = v[0].as_string();
            let a1 = v[1].as_char();
            a0.push(a1).into()
        })
        .f("push_string", |ctx, t, v| {
            let a0 = v[0].as_string();
            let a1 = v[1].as_string();
            a0.push_string(a1).into()
        })
        .f("remove", |ctx, t, v| {
            let a0 = v[0].as_string();
            let a1 = v[1].as_usize();
            let (a, b) = a0.remove(a1);
            Tuple(vector![a.into(), b.into()]).into()
        })
        .f("insert_char", |ctx, t, v| {
            let a0 = v[0].as_string();
            let a1 = v[1].as_usize();
            let a2 = v[2].as_char();
            a0.insert(a1, a2).into()
        })
        .f("is_empty", |ctx, t, v| {
            let a0 = v[0].as_string();
            a0.is_empty().into()
        })
        .f("split_off", |ctx, t, v| {
            let a0 = v[0].as_string();
            let a1 = v[1].as_usize();
            let (a, b) = a0.split_off(a1);
            Tuple(vector![a.into(), b.into()]).into()
        })
        .f("clear", |ctx, t, v| {
            let a0 = v[0].as_string();
            a0.clear().into()
        })
        .f("len", |ctx, t, v| {
            let a0 = v[0].as_string();
            a0.len().into()
        })
        .f("lines", |ctx, t, v| {
            let a0 = v[0].as_string();
            let a0: Vec<_> = a0
                .as_ref()
                .lines()
                .map(|x| String::from(x).into())
                .collect::<std::vec::Vec<Value>>()
                .into();
            a0.into()
        })
        .f("decode", |ctx, t, v| {
            todo!()
            // let a0 = a[0].as_string().0;
            // let a1 = a[1].as_encoding().0;
            // a0.decode(a1)
        })
        .f("encode", |ctx, t, v| {
            todo!()
            // let a0 = a[0].as_string().0;
            // a0.encode(a[1].as_encoding())
        });
}
