use hir::Type;
use serde::Deserialize;
use serde::Serialize;

use crate::Value;
use ast::binop;
use ast::unop;
use builtins::string::String;

pub fn define(builder: &mut super::Bifs) {
    builder
        .f(binop!(+), |ctx, t, v| {
            let v0 = v[0].as_i32();
            let v1 = v[1].as_i32();
            (v0 + v1).into()
        })
        .f(binop!(-), |ctx, t, v| {
            let v0 = v[0].as_i32();
            let v1 = v[1].as_i32();
            (v0 - v1).into()
        })
        .f(binop!(*), |ctx, t, v| {
            let v0 = v[0].as_i32();
            let v1 = v[1].as_i32();
            (v0 * v1).into()
        })
        .f(binop!(/), |ctx, t, v| {
            let v0 = v[0].as_i32();
            let v1 = v[1].as_i32();
            (v0 / v1).into()
        })
        .f(binop!(==), |ctx, t, v| {
            let v0 = v[0].as_i32();
            let v1 = v[1].as_i32();
            (v0 == v1).into()
        })
        .f(binop!(!=), |ctx, t, v| {
            let v0 = v[0].as_i32();
            let v1 = v[1].as_i32();
            (v0 != v1).into()
        })
        .f(binop!(<), |ctx, t, v| {
            let v0 = v[0].as_i32();
            let v1 = v[1].as_i32();
            (v0 < v1).into()
        })
        .f(binop!(<=), |ctx, t, v| {
            let v0 = v[0].as_i32();
            let v1 = v[1].as_i32();
            (v0 <= v1).into()
        })
        .f(binop!(>), |ctx, t, v| {
            let v0 = v[0].as_i32();
            let v1 = v[1].as_i32();
            (v0 > v1).into()
        })
        .f(binop!(>=), |ctx, t, v| {
            let v0 = v[0].as_i32();
            let v1 = v[1].as_i32();
            (v0 >= v1).into()
        })
        .f(unop!(-), |ctx, t, v| {
            let v0 = v[0].as_i32();
            (-v0).into()
        })
        .f(unop!(+), |ctx, t, v| {
            let v0 = v[0].as_i32();
            (v0).into()
        })
        .f("as_usize", |ctx, t, v| {
            let v0 = v[0].as_i32();
            (v0 as usize).into()
        })
        .f("i32_to_string", |ctx, t, v| {
            let v0 = v[0].as_i32();
            String::from(v0.to_string()).into()
        });
}
