use std::error::Error;
use std::fmt::Write;
use std::io::Result;

use value::dynamic::Instance;

use crate::definitions::*;

use super::*;
use value::dynamic::Function;

pub fn define(builder: &mut super::Bifs) {
    builder
        .f("logpath", |ctx, t, v| {
            let v0 = v[0].as_instance();
            v0.log.into()
        })
        .f("kill", |ctx, t, v| todo!());
}
