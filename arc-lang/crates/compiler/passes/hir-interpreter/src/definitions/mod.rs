use std::collections::HashMap;

use crate::context::Context;
use crate::Value;
use ast::Attr;
use ast::Bound;
use ast::Const;
use ast::Const::CString;
use ast::Generic;
use ast::Meta;
use ast::Name;
use ast::SBuiltinType;
use ast::StmtKind::SBuiltinClass;
use ast::StmtKind::SBuiltinDef;
use ast::StmtKind::SBuiltinInstance;
use ast::TypeKind::TArray;
use ast::TypeKind::TName;
use hir::Type;
use im_rc::vector;
use im_rc::Vector;
use info::Info;

pub mod aggregator;
pub mod array;
pub mod blob;
pub mod bool;
pub mod char;
pub mod dataflow;
pub mod discretizer;
pub mod duration;
pub mod encoding;
pub mod f32;
pub mod f64;
pub mod file;
pub mod function;
pub mod i16;
pub mod i32;
pub mod i64;
pub mod i8;
pub mod instance;
pub mod keyed_stream;
pub mod matrix;
pub mod model;
pub mod never;
pub mod option;
pub mod path;
pub mod reader;
pub mod record;
pub mod result;
pub mod socket;
pub mod stream;
pub mod string;
pub mod time;
pub mod time_source;
pub mod tuple;
pub mod u16;
pub mod u32;
pub mod u64;
pub mod u8;
pub mod unit;
pub mod url;
pub mod usize;
pub mod variant;
pub mod vec;
pub mod writer;

#[derive(Clone, Debug)]
pub struct Bifs(pub HashMap<&'static str, fn(&mut Context, &[Type], &[Value]) -> Value>);

impl Bifs {
    pub fn new() -> Self {
        let mut this = Self(HashMap::new());
        this.load(aggregator::define)
            .load(array::define)
            .load(blob::define)
            .load(bool::define)
            .load(char::define)
            .load(dataflow::define)
            .load(discretizer::define)
            .load(duration::define)
            .load(encoding::define)
            .load(f32::define)
            .load(f64::define)
            .load(file::define)
            .load(function::define)
            .load(i16::define)
            .load(i32::define)
            .load(i64::define)
            .load(i8::define)
            .load(instance::define)
            .load(keyed_stream::define)
            .load(matrix::define)
            .load(model::define)
            .load(option::define)
            .load(path::define)
            .load(reader::define)
            .load(record::define)
            .load(result::define)
            .load(socket::define)
            .load(stream::define)
            .load(string::define)
            .load(time::define)
            .load(time_source::define)
            .load(tuple::define)
            .load(u16::define)
            .load(u32::define)
            .load(u64::define)
            .load(u8::define)
            .load(unit::define)
            .load(url::define)
            .load(usize::define)
            .load(variant::define)
            .load(vec::define)
            .load(writer::define);
        this
    }

    fn load(&mut self, f: fn(&mut Self)) -> &mut Self {
        f(self);
        self
    }

    pub(crate) fn f(
        &mut self,
        name: &'static str,
        f: fn(&mut Context, &[Type], &[Value]) -> Value,
    ) -> &mut Self {
        self.0.insert(name, f);
        self
    }

    pub(crate) fn get(&self, name: &str) -> fn(&mut Context, &[Type], &[Value]) -> Value {
        if let Some(f) = self.0.get(name) {
            *f
        } else {
            panic!("builtin function {} not found", name)
        }
    }
}
