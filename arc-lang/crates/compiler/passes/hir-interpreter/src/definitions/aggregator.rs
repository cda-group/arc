use std::io::Result;
use std::io::Write;
use std::rc::Rc;

use builtins::aggregator::Aggregator;
use hir::Name;
use hir::Type;
use serde::Deserialize;
use serde::Serialize;

use crate::definitions::*;

use value::dynamic::Function;
use super::*;

pub fn define(builder: &mut super::Bifs) {
    builder.f("aggregator", |ctx, t, v| {
        let a0 = v[0].as_function();
        let a1 = v[1].as_function();
        let a2 = v[2].as_function();
        let a3 = v[3].as_function();
        Aggregator::Monoid {
            lift: a0,
            combine: a1,
            identity: a2,
            lower: a3,
        }
        .into()
    });
}
