use std::cell::RefCell;
use std::rc::Rc;

use ast::Name;
use builtins::aggregator::Aggregator;
use builtins::discretizer::Discretizer;
use builtins::encoding::Encoding;
use builtins::path::Path;
use builtins::reader::Reader;
use builtins::time_source::TimeSource;
use builtins::writer::Writer;
use im_rc::vector;
use im_rc::HashMap;
use im_rc::OrdMap;
use im_rc::OrdSet;
use im_rc::Vector;
use serde::Deserialize;
use serde::Serialize;

use crate::Value;

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize)]
pub struct Array(pub Vector<Value>);

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct Function(pub Name);

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Matrix {
    I8(builtins::matrix::Matrix<i8>),
    I16(builtins::matrix::Matrix<i16>),
    I32(builtins::matrix::Matrix<i32>),
    I64(builtins::matrix::Matrix<i64>),
    U8(builtins::matrix::Matrix<u8>),
    U16(builtins::matrix::Matrix<u16>),
    U32(builtins::matrix::Matrix<u32>),
    U64(builtins::matrix::Matrix<u64>),
    F32(builtins::matrix::Matrix<f32>),
    F64(builtins::matrix::Matrix<f64>),
}

impl Matrix {
    fn as_i8(self) -> builtins::matrix::Matrix<i8> {
        match self {
            Matrix::I8(v) => v,
            _ => unreachable!(),
        }
    }
    fn as_i16(self) -> builtins::matrix::Matrix<i16> {
        match self {
            Matrix::I16(v) => v,
            _ => unreachable!(),
        }
    }
    fn as_i32(self) -> builtins::matrix::Matrix<i32> {
        match self {
            Matrix::I32(v) => v,
            _ => unreachable!(),
        }
    }
    fn as_i64(self) -> builtins::matrix::Matrix<i64> {
        match self {
            Matrix::I64(v) => v,
            _ => unreachable!(),
        }
    }
    fn as_u8(self) -> builtins::matrix::Matrix<u8> {
        match self {
            Matrix::U8(v) => v,
            _ => unreachable!(),
        }
    }
    fn as_u16(self) -> builtins::matrix::Matrix<u16> {
        match self {
            Matrix::U16(v) => v,
            _ => unreachable!(),
        }
    }
    fn as_u32(self) -> builtins::matrix::Matrix<u32> {
        match self {
            Matrix::U32(v) => v,
            _ => unreachable!(),
        }
    }
    fn as_u64(self) -> builtins::matrix::Matrix<u64> {
        match self {
            Matrix::U64(v) => v,
            _ => unreachable!(),
        }
    }
    fn as_f32(self) -> builtins::matrix::Matrix<f32> {
        match self {
            Matrix::F32(v) => v,
            _ => unreachable!(),
        }
    }
    fn as_f64(self) -> builtins::matrix::Matrix<f64> {
        match self {
            Matrix::F64(v) => v,
            _ => unreachable!(),
        }
    }
}

#[macro_export]
macro_rules! map_matrix {
    { $v:expr, $f:expr } => {
        match $v {
            Matrix::I8(v) => $f(v),
            Matrix::I16(v) => $f(v),
            Matrix::I32(v) => $f(v),
            Matrix::I64(v) => $f(v),
            Matrix::U8(v) => $f(v),
            Matrix::U16(v) => $f(v),
            Matrix::U32(v) => $f(v),
            Matrix::U64(v) => $f(v),
            Matrix::F32(v) => $f(v),
            Matrix::F64(v) => $f(v),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize)]
pub struct Record(pub HashMap<Name, Value>);

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize)]
pub struct Tuple(pub Vector<Value>);

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize)]
pub struct Variant {
    pub x: Name,
    pub v: Value,
}

#[derive(Debug, Clone)]
pub struct Stream {
    pub prefix: Vector<Stream>,
    pub name: Name,
    pub kind: Rc<StreamKind>,
}

impl Stream {
    pub fn extend(self, name: Name, kind: StreamKind) -> Self {
        let mut prefix = self.prefix.clone();
        prefix.push_back(self);
        Self { prefix, name, kind: Rc::new(kind) }
    }

    pub fn new(name: Name, kind: StreamKind) -> Self {
        Self {
            prefix: vector![],
            name,
            kind: Rc::new(kind),
        }
    }
}

pub use StreamKind::*;
#[derive(Debug, Clone)]
pub enum StreamKind {
    DSource(Reader, Encoding, TimeSource<Function>),
    DMap(Name, Function),
    DFilter(Name, Function),
    DFlatten(Name),
    DFlatMap(Name, Function),
    DScan(Name, Function),
    DKeyby(Name, Function),
    DUnkey(Name),
    DApply(Name, Function),
    DWindow(Name, Discretizer, Aggregator<Function, Function, Function, Function>),
    DMerge(Vector<Name>),
}

#[derive(Debug, Clone)]
pub struct Dataflow {
    pub streams: Vector<Stream>,
    pub sinks: Vector<Sink>,
}

#[derive(Debug, Clone)]
pub struct Instance {
    pub log: Path,
}

impl Dataflow {
    pub fn new(streams: Vector<Stream>, sinks: Vector<Sink>) -> Self {
        Self { streams, sinks }
    }
}

#[derive(Debug, Clone)]
pub struct Sink(pub Rc<(Name, Writer, Encoding)>);

impl Sink {
    pub fn new(stream: Name, writer: Writer, encoding: Encoding) -> Self {
        Self(Rc::new((stream, writer, encoding)))
    }
}
