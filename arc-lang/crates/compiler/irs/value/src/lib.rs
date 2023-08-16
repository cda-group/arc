#![allow(unused)]

pub mod conv;
pub mod de;
pub mod dynamic;
pub mod eq;
pub mod hash;
pub mod ord;
pub mod ser;

use std::cell::RefCell;
use std::rc::Rc;

use builtins::aggregator::Aggregator;
use builtins::blob::Blob;
use builtins::dict::Dict;
use builtins::discretizer::Discretizer;
use builtins::duration::Duration;
use builtins::encoding::Encoding;
use builtins::file::File;
use builtins::model::Model;
use builtins::path::Path;
use builtins::reader::Reader;
use builtins::set::Set;
use builtins::socket::SocketAddr;
use builtins::time::Time;
use builtins::time_source::TimeSource;
use builtins::url::Url;
use builtins::writer::Writer;
use dynamic::Array;
use dynamic::Dataflow;
use dynamic::Function;
use dynamic::Instance;
use dynamic::Matrix;
use dynamic::Record;
use dynamic::Stream;
use dynamic::Tuple;
use dynamic::Variant;
use hir::Name;
use im_rc::HashMap;
use im_rc::Vector;
use serde::Deserialize;
use serde::Serialize;
pub use ValueKind::*;

#[derive(Clone)]
pub struct Value {
    pub kind: Rc<ValueKind>,
}

impl From<ValueKind> for Value {
    fn from(kind: ValueKind) -> Self {
        Value::new(kind)
    }
}

impl Value {
    pub fn new(kind: ValueKind) -> Value {
        Value { kind: Rc::new(kind) }
    }
}

#[derive(Clone)]
pub enum ValueKind {
    VAggregator(Aggregator<Function, Function, Function, Function>),
    VArray(Array),
    VBlob(Blob),
    VBool(bool),
    VChar(char),
    VDict(Dict<Value, Value>),
    VDiscretizer(Discretizer),
    VDuration(Duration),
    VEncoding(Encoding),
    VF32(f32),
    VF64(f64),
    VFile(File),
    VFunction(Function),
    VI128(i128),
    VI16(i16),
    VI32(i32),
    VI64(i64),
    VI8(i8),
    VMatrix(Matrix),
    VModel(Model),
    VOption(builtins::option::Option<Value>),
    VPath(Path),
    VReader(Reader),
    VRecord(Record),
    VResult(builtins::result::Result<Value>),
    VSet(Set<Value>),
    VSocketAddr(SocketAddr),
    VStream(Stream),
    VDataflow(Dataflow),
    VString(builtins::string::String),
    VTime(Time),
    VTimeSource(TimeSource<Function>),
    VTuple(Tuple),
    VU128(u128),
    VU16(u16),
    VU32(u32),
    VU64(u64),
    VU8(u8),
    VUnit(()),
    VUrl(Url),
    VUsize(usize),
    VVariant(Variant),
    VVec(builtins::vec::Vec<Value>),
    VWriter(Writer),
    VInstance(Instance),
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind.as_ref() {
            VAggregator(v) => v.fmt(f),
            VArray(v) => v.fmt(f),
            VBlob(v) => v.fmt(f),
            VBool(v) => v.fmt(f),
            VChar(v) => v.fmt(f),
            VDict(v) => v.fmt(f),
            VDiscretizer(v) => v.fmt(f),
            VDuration(v) => v.fmt(f),
            VEncoding(v) => v.fmt(f),
            VF32(v) => v.fmt(f),
            VF64(v) => v.fmt(f),
            VFile(v) => v.fmt(f),
            VFunction(v) => v.fmt(f),
            VI128(v) => v.fmt(f),
            VI16(v) => v.fmt(f),
            VI32(v) => v.fmt(f),
            VI64(v) => v.fmt(f),
            VI8(v) => v.fmt(f),
            VMatrix(v) => v.fmt(f),
            VModel(v) => v.fmt(f),
            VOption(v) => v.fmt(f),
            VPath(v) => v.fmt(f),
            VReader(v) => v.fmt(f),
            VRecord(v) => v.fmt(f),
            VResult(v) => v.fmt(f),
            VSet(v) => v.fmt(f),
            VSocketAddr(v) => v.fmt(f),
            VStream(v) => v.fmt(f),
            VDataflow(v) => v.fmt(f),
            VString(v) => v.fmt(f),
            VTime(v) => v.fmt(f),
            VTimeSource(v) => v.fmt(f),
            VTuple(v) => v.fmt(f),
            VU128(v) => v.fmt(f),
            VU16(v) => v.fmt(f),
            VU32(v) => v.fmt(f),
            VU64(v) => v.fmt(f),
            VU8(v) => v.fmt(f),
            VUnit(v) => v.fmt(f),
            VUrl(v) => v.fmt(f),
            VUsize(v) => v.fmt(f),
            VVariant(v) => v.fmt(f),
            VVec(v) => v.fmt(f),
            VWriter(v) => v.fmt(f),
            VInstance(v) => v.fmt(f),
        }
    }
}

#[cfg(test)]
mod test {
    use im_rc::hashmap;
    use im_rc::vector;
    use serde::de::DeserializeSeed;
    use serde::Serialize;
    use serde_json::de::StrRead;

    use crate::de::Seed;
    use crate::Record;
    use crate::Value;

    #[test]
    fn serde_i32() {
        let v0 = Value::from(1);
        let s = serde_json::to_string(&v0).unwrap();
        let mut de = serde_json::Deserializer::from_str(&s);
        let t = hir::TNominal("i32".to_string(), vector![]).into();
        let v1 = Seed(t).deserialize(&mut de).unwrap();
        assert_eq!(v0, v1);
        assert_eq!(s, "1");
    }

    #[test]
    fn serde_vec() {
        let v0 = Value::from(1);
        let v1 = Value::from(2);
        let v2 = Value::from(3);
        let v3 = Value::from(builtins::vec::Vec::from(vec![v0, v1, v2]));
        let s = serde_json::to_string(&v3).unwrap();
        let mut de = serde_json::Deserializer::from_str(&s);
        let t0 = hir::TNominal("i32".to_string(), vector![]).into();
        let t1 = hir::TNominal("Vec".to_string(), vector![t0]).into();
        let v4 = Seed(t1).deserialize(&mut de).unwrap();
        assert_eq!(v3, v4);
        assert_eq!(s, "[1,2,3]");
    }

    #[test]
    fn serde_tuple() {
        let v0 = Value::from(1);
        let v1 = Value::from(2);
        let v2 = Value::from(builtins::string::String::from("Hello"));
        let v3 = Value::from(crate::dynamic::Tuple(vector![v0, v1, v2]));
        let s = serde_json::to_string(&v3).unwrap();
        let mut de = serde_json::Deserializer::from_str(&s);
        let t0 = hir::TNominal("i32".to_string(), vector![]).into();
        let t1 = hir::TNominal("i32".to_string(), vector![]).into();
        let t2 = hir::TNominal("String".to_string(), vector![]).into();
        let t3 = hir::TTuple(vector![t0, t1, t2], true).into();
        let v4 = Seed(t3).deserialize(&mut de).unwrap();
        assert_eq!(v3, v4);
        assert_eq!(s, r#"[1,2,"Hello"]"#);
    }

    #[test]
    fn serde_record() {
        let v0 = Value::from(1);
        let v1 = Value::from(builtins::string::String::from("Hello"));
        let v2 = Value::from(Record(hashmap! {
            "a".to_string() => v0,
            "b".to_string() => v1,
        }));
        let s = serde_json::to_string(&v2).unwrap();
        let mut de = serde_json::Deserializer::from_str(&s);
        let t0 = hir::TNominal("i32".to_string(), vector![]).into();
        let t1 = hir::TNominal("String".to_string(), vector![]).into();
        let t2 = hir::TRecord(hir::fields_to_row(vector![("a".to_string(), t0), ("b".to_string(), t1),])).into();
        let v3 = Seed(t2).deserialize(&mut de).unwrap();
        assert_eq!(v2, v3);
        assert!((s == r#"{"a":1,"b":"Hello"}"#) || (s == r#"{"b":"Hello","a":1}"#));
    }

    #[test]
    fn serde_dict() {
        let k0 = Value::from(builtins::string::String::from("a"));
        let k1 = Value::from(builtins::string::String::from("b"));
        let v0 = Value::from(1);
        let v1 = Value::from(2);
        let v2 = Value::from(builtins::dict::Dict::from(vec![(k0, v0), (k1, v1)].into_iter().collect::<std::collections::HashMap<_, _>>()));
        let s = serde_json::to_string(&v2).unwrap();
        let mut de = serde_json::Deserializer::from_str(&s);
        let t0 = hir::TNominal("String".to_string(), vector![]).into();
        let t1 = hir::TNominal("i32".to_string(), vector![]).into();
        let t2 = hir::TNominal("Dict".to_string(), vector![t0, t1]).into();
        let v3 = Seed(t2).deserialize(&mut de).unwrap();
        assert_eq!(v2, v3);
        assert!((s == r#"{"a":1,"b":2}"#) || (s == r#"{"b":2,"a":1}"#));
    }

    #[test]
    fn serde_array() {
        let v0 = Value::from(1);
        let v1 = Value::from(2);
        let v2 = Value::from(3);
        let v3 = Value::from(crate::dynamic::Array(vector![v0, v1, v2]));
        let s = serde_json::to_string(&v3).unwrap();
        let mut de = serde_json::Deserializer::from_str(&s);
        let t0 = hir::TNominal("i32".to_string(), vector![]).into();
        let t1 = hir::TArray(t0, Some(3)).into();
        let v4 = Seed(t1).deserialize(&mut de).unwrap();
        assert_eq!(v3, v4);
        assert_eq!(s, "[1,2,3]");
    }

    #[test]
    fn serde_set() {
        let v0 = Value::from(1);
        let v1 = Value::from(2);
        let v2 = Value::from(builtins::set::Set::from(vec![v0, v1].into_iter().collect::<std::collections::HashSet<_>>()));
        let s = serde_json::to_string(&v2).unwrap();
        let mut de = serde_json::Deserializer::from_str(&s);
        let t0 = hir::TNominal("i32".to_string(), vector![]).into();
        let t1 = hir::TNominal("Set".to_string(), vector![t0]).into();
        let v3 = Seed(t1).deserialize(&mut de).unwrap();
        assert_eq!(v2, v3);
        assert!((s == r#"[1,2]"#) || (s == r#"[2,1]"#));
    }

    #[test]
    fn serde_option_some() {
        let v0 = Value::from(1);
        let v1 = Value::from(builtins::option::Option::some(v0));
        let s = serde_json::to_string(&v1).unwrap();
        let mut de = serde_json::Deserializer::from_str(&s);
        let t0 = hir::TNominal("i32".to_string(), vector![]).into();
        let t1 = hir::TNominal("Option".to_string(), vector![t0]).into();
        let v2 = Seed(t1).deserialize(&mut de).unwrap();
        assert_eq!(v1, v2);
        assert_eq!(s, "1");
    }

    #[test]
    fn serde_option_none() {
        let v0 = Value::from(builtins::option::Option::none());
        let s = serde_json::to_string(&v0).unwrap();
        let mut de = serde_json::Deserializer::from_str(&s);
        let t0 = hir::TNominal("i32".to_string(), vector![]).into();
        let t1 = hir::TNominal("Option".to_string(), vector![t0]).into();
        let v2 = Seed(t1).deserialize(&mut de).unwrap();
        assert_eq!(v0, v2);
        assert_eq!(s, "null");
    }

    #[test]
    fn serde_result_ok() {
        let v0 = Value::from(1);
        let v1 = Value::from(builtins::result::Result::ok(v0));
        let s = serde_json::to_string(&v1).unwrap();
        let mut de = serde_json::Deserializer::from_str(&s);
        let t0 = hir::TNominal("i32".to_string(), vector![]).into();
        let t1 = hir::TNominal("Result".to_string(), vector![t0]).into();
        let v2 = Seed(t1).deserialize(&mut de).unwrap();
        assert_eq!(v1, v2);
        assert_eq!(s, r#"{"Ok":1}"#);
    }

    #[test]
    fn serde_result_err() {
        let v0 = builtins::string::String::from("Hello");
        let v1 = Value::from(builtins::result::Result::error(v0));
        let s = serde_json::to_string(&v1).unwrap();
        let mut de = serde_json::Deserializer::from_str(&s);
        let t0 = hir::TNominal("i32".to_string(), vector![]).into();
        let t1 = hir::TNominal("Result".to_string(), vector![t0]).into();
        let v2 = Seed::deserialize(Seed(t1), &mut de).unwrap();
        assert_eq!(v1, v2);
        assert_eq!(s, r#"{"Err":"Hello"}"#);
    }

    #[test]
    fn serde_matrix() {
        let v9 = Value::from(crate::dynamic::Matrix::I32(builtins::matrix::Matrix::new([2, 2])));
        let s = serde_json::to_string(&v9).unwrap();
        let mut de = serde_json::Deserializer::from_str(&s);
        let t0 = hir::TNominal("i32".to_string(), vector![]).into();
        let t1 = hir::TNominal("Matrix".to_string(), vector![t0]).into();
        let v10 = Seed(t1).deserialize(&mut de).unwrap();
        assert_eq!(v9, v10);
        assert_eq!(s, r#"{"v":1,"dim":[2,2],"data":[0,0,0,0]}"#);
    }

    #[test]
    fn serde_type_variable() {
        let mut de = serde_json::Deserializer::from_str("1");
        let t = hir::TVar("a".to_string()).into();
        assert!(Seed(t).deserialize(&mut de).is_err());
    }

    #[test]
    fn serde_type_error() {
        let mut de = serde_json::Deserializer::from_str("1.0");
        let t = hir::TNominal("i32".to_string(), vector![]).into();
        assert!(Seed(t).deserialize(&mut de).is_err());
    }
}
