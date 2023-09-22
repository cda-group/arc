use std::collections::HashMap;

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
use ast::Type;
use ast::TypeKind::TArray;
use ast::TypeKind::TFun;
use ast::TypeKind::TName;
use ast::TypeKind::TTuple;
use ast::TypeKind::TUnit;
use im_rc::vector;
use im_rc::Vector;
use info::Info;

pub mod aggregator;
pub mod array;
pub mod blob;
pub mod bool;
pub mod char;
pub mod dataflow;
pub mod dict;
pub mod discretizer;
pub mod duration;
pub mod encoding;
pub mod f32;
pub mod f64;
pub mod file;
pub mod function;
pub mod i128;
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
pub mod u128;
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

pub fn prelude() -> Vector<ast::Stmt> {
    Builder::new()
        .load(aggregator::declare)
        .load(array::declare)
        .load(blob::declare)
        .load(bool::declare)
        .load(char::declare)
        .load(dataflow::declare)
        .load(discretizer::declare)
        .load(duration::declare)
        .load(encoding::declare)
        .load(f32::declare)
        .load(f64::declare)
        .load(file::declare)
        .load(function::declare)
        .load(i16::declare)
        .load(i32::declare)
        .load(i64::declare)
        .load(i8::declare)
        .load(instance::declare)
        .load(keyed_stream::declare)
        .load(matrix::declare)
        .load(model::declare)
        .load(option::declare)
        .load(path::declare)
        .load(reader::declare)
        .load(record::declare)
        .load(result::declare)
        .load(socket::declare)
        .load(stream::declare)
        .load(string::declare)
        .load(time::declare)
        .load(time_source::declare)
        .load(tuple::declare)
        .load(u16::declare)
        .load(u32::declare)
        .load(u64::declare)
        .load(u8::declare)
        .load(unit::declare)
        .load(url::declare)
        .load(usize::declare)
        .load(variant::declare)
        .load(vec::declare)
        .load(writer::declare)
        .load(i128::declare)
        .load(u128::declare)
        .build()
}

fn t(name: &str) -> Type {
    TName(name.into(), vector![]).with(Info::Builtin)
}

fn tc<const N: usize>(name: &str, args: [Type; N]) -> Type {
    TName(name.into(), args.into_iter().collect()).with(Info::Builtin)
}

pub(crate) struct Builder {
    types: Vec<ast::Stmt>,
    functions: Vec<ast::Stmt>,
}

pub fn mlir(s: &str) -> (&str, Option<Const>) {
    ("mlir", Some(CString(s.to_owned())))
}

pub fn rust(s: &str) -> (&str, Option<Const>) {
    ("rust", Some(CString(s.to_owned())))
}

pub fn noop() -> &'static str {
    "(|| ())"
}

fn meta<const N: usize>(attrs: [(&str, Option<Const>); N]) -> Vector<Attr> {
    attrs
        .into_iter()
        .map(|(x, c)| Attr {
            x: x.to_string(),
            c,
            info: Info::Builtin,
        })
        .collect()
}

impl Builder {
    fn new() -> Self {
        Self { types: vec![], functions: vec![] }
    }

    fn load(&mut self, f: impl Fn(&mut Self)) -> &mut Self {
        f(self);
        self
    }

    fn t<const N: usize, const M: usize>(&mut self, x: &str, gs: [&str; M], attrs: [(&str, Option<Const>); N]) -> &mut Self {
        self.types
            .push(SBuiltinType(meta(attrs), x.into(), gs.into_iter().map(Into::into).collect(), vector![]).with(Info::Builtin));
        self
    }

    // fn class<const N: usize, const M: usize>(&mut self, x: &str, gs: [&str; N], bs: [Bound; M]) -> &mut Self {
    //     self.prelude
    //         .push(SBuiltinClass(vector![], x.into(), gs.into_iter().map(Into::into).collect(), bs.into_iter().collect()).with(Info::Builtin));
    //     self
    // }

    // fn instance<const N: usize, const M: usize>(&mut self, x: &str, gs: [&str; N], bs: [Bound; M], t: Type) -> &mut Self {
    //     self.prelude
    //         .push(SBuiltinInstance(vector![], x.into(), gs.into_iter().map(Into::into).collect(), bs.into_iter().map(Into::into).collect(), t).with(Info::Builtin));
    //     self
    // }

    fn f<const N: usize, const M: usize, const K: usize>(&mut self, x: &str, gs: [&str; M], ts: [Type; K], t: Type, attrs: [(&str, Option<Const>); N]) -> &mut Self {
        let t = if matches!(t.kind.as_ref(), TUnit) { None } else { Some(t) };
        self.functions
            .push(SBuiltinDef(meta(attrs), x.into(), gs.into_iter().map(Into::into).collect(), ts.into_iter().collect(), t, vector![]).with(Info::Builtin));
        self
    }

    fn build(&mut self) -> Vector<ast::Stmt> {
        let mut prelude = std::mem::take(&mut self.types);
        prelude.append(&mut self.functions);
        prelude.into()
    }
}
