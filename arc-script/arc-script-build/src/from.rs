//! Module for converting staged inputs into HIR values.
use crate::partial::Partial;
use crate::partial::PartialKind;
use arc_script_core::prelude::{Value, ValueKind};

impl From<Partial> for Value {
    #[rustfmt::skip]
    fn from(input: Partial) -> Self {
        let kind = match input.kind {
            PartialKind::I8(v)      => ValueKind::I8(v),
            PartialKind::I16(v)     => ValueKind::I16(v),
            PartialKind::I32(v)     => ValueKind::I32(v),
            PartialKind::I64(v)     => ValueKind::I64(v),
            PartialKind::U8(v)      => ValueKind::U8(v),
            PartialKind::U16(v)     => ValueKind::U16(v),
            PartialKind::U32(v)     => ValueKind::U32(v),
            PartialKind::U64(v)     => ValueKind::U64(v),
            PartialKind::Bf16(v)    => ValueKind::Bf16(v),
            PartialKind::F16(v)     => ValueKind::F16(v),
            PartialKind::F32(v)     => ValueKind::F32(v),
            PartialKind::F64(v)     => ValueKind::F64(v),
            PartialKind::Char(v)    => ValueKind::Char(v),
            PartialKind::Str(v)     => ValueKind::Str(v),
            PartialKind::Bool(v)    => ValueKind::Bool(v),
            PartialKind::Vector(v)  => {
                ValueKind::Vector(v.0.into_iter().map(|v| Value::from(v)).collect::<Vec<_>>())
            }
            PartialKind::Array(v)   => todo!(),
            PartialKind::Variant(v) => todo!(),
            PartialKind::Struct(v)  => todo!(),
            PartialKind::Tuple(v)   => todo!(),
        };
        todo!()
    }
}
