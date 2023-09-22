use hir::Type;
use serde::de::DeserializeSeed;
use serde::de::MapAccess;
use serde::de::VariantAccess;
use serde::de::Visitor;
use serde::Deserializer;

use crate::*;

pub struct Seed(pub Type);

impl<'de> DeserializeSeed<'de> for Seed {
    type Value = Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        match self.0.kind.as_ref() {
            hir::TFun(_, _) => unreachable!(),
            hir::TTuple(ts, _) => {
                struct TupleVisitor(Vector<Type>);
                impl<'de> Visitor<'de> for TupleVisitor {
                    type Value = Value;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                        write!(formatter, "a tuple of length {}", self.0.len())
                    }

                    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
                    where
                        A: serde::de::SeqAccess<'de>,
                    {
                        let mut v = Vector::new();
                        for t in self.0 {
                            let context = Seed(t);
                            v.push_back(seq.next_element_seed(context)?.unwrap());
                        }
                        Ok(Value::from(Tuple(v)))
                    }
                }
                deserializer.deserialize_tuple(ts.len(), TupleVisitor(ts.clone()))
            }
            hir::TRecord(t) => {
                struct RecordVisitor(HashMap<Name, Type>);
                impl<'de> Visitor<'de> for RecordVisitor {
                    type Value = Value;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                        write!(formatter, "a record with fields {:?}", self.0)
                    }

                    fn visit_map<A>(mut self, mut map: A) -> Result<Self::Value, A::Error>
                    where
                        A: MapAccess<'de>,
                    {
                        let mut result = HashMap::new();
                        while !self.0.is_empty() {
                            let k = map.next_key()?.unwrap();
                            if let Some(t) = self.0.remove(&k) {
                                result.insert(k, map.next_value_seed(Seed(t))?);
                            } else {
                                return Err(serde::de::Error::custom("Found unexpected field"));
                            }
                        }
                        Ok(Value::from(Record(result)))
                    }
                }
                deserializer.deserialize_map(RecordVisitor(
                    hir::row_to_fields(t.clone()).into_iter().collect(),
                ))
            }
            hir::TNominal(x, ts) => match x.as_str() {
                "i8" => i8::deserialize(deserializer).map(Value::from),
                "i16" => i16::deserialize(deserializer).map(Value::from),
                "i32" => i32::deserialize(deserializer).map(Value::from),
                "i64" => i64::deserialize(deserializer).map(Value::from),
                "u8" => u8::deserialize(deserializer).map(Value::from),
                "u16" => u16::deserialize(deserializer).map(Value::from),
                "u32" => u32::deserialize(deserializer).map(Value::from),
                "u64" => u64::deserialize(deserializer).map(Value::from),
                "usize" => usize::deserialize(deserializer).map(Value::from),
                "f32" => f32::deserialize(deserializer).map(Value::from),
                "f64" => f64::deserialize(deserializer).map(Value::from),
                "bool" => bool::deserialize(deserializer).map(Value::from),
                "char" => char::deserialize(deserializer).map(Value::from),
                "String" => String::deserialize(deserializer)
                    .map(builtins::string::String::from)
                    .map(Value::from),
                "Dict" => {
                    struct DictVisitor(Type, Type);
                    impl<'de> Visitor<'de> for DictVisitor {
                        type Value = Value;

                        fn expecting(
                            &self,
                            formatter: &mut std::fmt::Formatter,
                        ) -> std::fmt::Result {
                            write!(formatter, "a dict")
                        }

                        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
                        where
                            A: MapAccess<'de>,
                        {
                            let mut result = std::collections::HashMap::new();
                            while let Some((k, v)) =
                                map.next_entry_seed(Seed(self.0.clone()), Seed(self.1.clone()))?
                            {
                                result.insert(k, v);
                            }
                            Ok(Value::from(builtins::dict::Dict::from(result)))
                        }
                    }
                    let k = ts[0].clone();
                    let v = ts[1].clone();
                    deserializer.deserialize_map(DictVisitor(k, v))
                }
                "Set" => {
                    struct SetVisitor(Type);
                    impl<'de> Visitor<'de> for SetVisitor {
                        type Value = Value;

                        fn expecting(
                            &self,
                            formatter: &mut std::fmt::Formatter,
                        ) -> std::fmt::Result {
                            write!(formatter, "a set")
                        }

                        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
                        where
                            A: serde::de::SeqAccess<'de>,
                        {
                            let mut result = std::collections::HashSet::new();
                            while let Some(v) = seq.next_element_seed(Seed(self.0.clone()))? {
                                result.insert(v);
                            }
                            Ok(Value::from(builtins::set::Set::from(result)))
                        }
                    }
                    let t = ts[0].clone();
                    deserializer.deserialize_seq(SetVisitor(t))
                }
                "Time" => builtins::time::Time::deserialize(deserializer).map(Value::from),
                "Duration" => {
                    builtins::duration::Duration::deserialize(deserializer).map(Value::from)
                }
                "Url" => builtins::url::Url::deserialize(deserializer).map(Value::from),
                "Path" => builtins::path::Path::deserialize(deserializer).map(Value::from),
                "Blob" => builtins::blob::Blob::deserialize(deserializer).map(Value::from),
                "Option" => {
                    struct OptionVisitor(Type);
                    impl<'de> Visitor<'de> for OptionVisitor {
                        type Value = Value;

                        fn expecting(
                            &self,
                            formatter: &mut std::fmt::Formatter,
                        ) -> std::fmt::Result {
                            write!(formatter, "an option")
                        }

                        fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
                        where
                            D: Deserializer<'de>,
                        {
                            let v = Seed(self.0).deserialize(deserializer)?;
                            Ok(Value::from(builtins::option::Option::some(v)))
                        }

                        fn visit_none<E>(self) -> Result<Self::Value, E> {
                            Ok(Value::from(builtins::option::Option::none()))
                        }
                    }
                    let t = ts[0].clone();
                    deserializer.deserialize_option(OptionVisitor(t))
                }
                "Result" => {
                    struct ResultVisitor(Type);
                    impl<'de> Visitor<'de> for ResultVisitor {
                        type Value = Value;

                        fn expecting(
                            &self,
                            formatter: &mut std::fmt::Formatter,
                        ) -> std::fmt::Result {
                            write!(formatter, "a result")
                        }

                        fn visit_enum<A>(self, data: A) -> Result<Self::Value, A::Error>
                        where
                            A: serde::de::EnumAccess<'de>,
                        {
                            let (v, variant) = data.variant()?;
                            println!("->{:?}", v);
                            match v {
                                "Ok" => {
                                    let v = variant.newtype_variant_seed(Seed(self.0.clone()))?;
                                    Ok(Value::from(builtins::result::Result::ok(v)))
                                }
                                "Err" => {
                                    let v = variant.newtype_variant()?;
                                    Ok(Value::from(builtins::result::Result::error(v)))
                                }
                                _ => unreachable!(),
                            }
                        }
                    }
                    let t = ts[0].clone();
                    deserializer.deserialize_enum("Result", &["Ok", "Err"], ResultVisitor(t))
                }
                "Vec" => {
                    struct VecVisitor(Type);
                    impl<'de> Visitor<'de> for VecVisitor {
                        type Value = Value;

                        fn expecting(
                            &self,
                            formatter: &mut std::fmt::Formatter,
                        ) -> std::fmt::Result {
                            write!(formatter, "a vec")
                        }

                        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
                        where
                            A: serde::de::SeqAccess<'de>,
                        {
                            let mut result = Vec::new();
                            while let Some(v) = seq.next_element_seed(Seed(self.0.clone()))? {
                                result.push(v);
                            }
                            Ok(Value::from(builtins::vec::Vec::from(result)))
                        }
                    }
                    let t = ts[0].clone();
                    deserializer.deserialize_seq(VecVisitor(t))
                }
                "Matrix" => {
                    let hir::TNominal(x, _) = ts[0].kind.as_ref() else {
                        unreachable!()
                    };
                    let m = match x.as_str() {
                        "i8" => {
                            Matrix::I8(builtins::matrix::Matrix::<i8>::deserialize(deserializer)?)
                        }
                        "i16" => {
                            Matrix::I16(builtins::matrix::Matrix::<i16>::deserialize(deserializer)?)
                        }
                        "i32" => {
                            Matrix::I32(builtins::matrix::Matrix::<i32>::deserialize(deserializer)?)
                        }
                        "i64" => {
                            Matrix::I64(builtins::matrix::Matrix::<i64>::deserialize(deserializer)?)
                        }
                        "u8" => {
                            Matrix::U8(builtins::matrix::Matrix::<u8>::deserialize(deserializer)?)
                        }
                        "u16" => {
                            Matrix::U16(builtins::matrix::Matrix::<u16>::deserialize(deserializer)?)
                        }
                        "u32" => {
                            Matrix::U32(builtins::matrix::Matrix::<u32>::deserialize(deserializer)?)
                        }
                        "u64" => {
                            Matrix::U64(builtins::matrix::Matrix::<u64>::deserialize(deserializer)?)
                        }
                        "f32" => {
                            Matrix::F32(builtins::matrix::Matrix::<f32>::deserialize(deserializer)?)
                        }
                        "f64" => {
                            Matrix::F64(builtins::matrix::Matrix::<f64>::deserialize(deserializer)?)
                        }
                        _ => unreachable!(),
                    };
                    Ok(Value::from(m))
                }
                _ => unreachable!("Attempted to deserialize undeserializable type {:?}", x),
            },
            hir::TAlias(_, _, _) => todo!(),
            hir::TRowEmpty => unreachable!(),
            hir::TRowExtend(_, _) => unreachable!(),
            hir::TRecordConcat(_, _) => unreachable!(),
            hir::TGeneric(_) => unreachable!(),
            hir::TArray(t, n) => {
                struct ArrayVisitor(Type);
                impl<'de> Visitor<'de> for ArrayVisitor {
                    type Value = Value;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                        write!(formatter, "an array")
                    }

                    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
                    where
                        A: serde::de::SeqAccess<'de>,
                    {
                        let mut result = Vector::new();
                        while let Some(v) = seq.next_element_seed(Seed(self.0.clone()))? {
                            result.push_back(v);
                        }
                        Ok(Value::from(Array(result)))
                    }
                }
                let n = n.unwrap() as usize;
                deserializer.deserialize_tuple(n, ArrayVisitor(t.clone()))
            }
            hir::TArrayConcat(_, _) => unreachable!(),
            hir::TUnit => <() as Deserialize>::deserialize(deserializer).map(Value::from),
            hir::TNever => unreachable!(),
            hir::TVar(_) => Err(serde::de::Error::custom(
                "Attempted to deserialize a type variable",
            )),
            hir::TError => unreachable!(),
        }
    }
}
