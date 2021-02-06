use arc_script_core::prelude::bf16;
use arc_script_core::prelude::f16;

use derive_more::From;

/// Arguments which can be passed to an arc-script when building it.
#[derive(Clone, Debug)]
pub enum Value {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    Bf16(bf16),
    F16(f16),
    F32(f32),
    F64(f64),
    Char(char),
    Str(String),
    Bool(bool),
    Vector(Box<Vector>),
    Array(Box<Array>),
    Variant(Box<Variant>),
    Struct(Struct),
    Tuple(Tuple),
}

macro_rules! from_primitive {
    {
        $variant:ident($ty:ty)
    } => {
        impl From<$ty> for Value {
            fn from(x: $ty) -> Self {
                Self::$variant(x)
            }
        }
    }
}

macro_rules! from_tuple {
    {
        $($x:ident),*
    } => {
        #[allow(non_camel_case_types)]
        impl<$($x),*> From<($($x),*,)> for Value
        where
            $($x: Into<Value>),*
        {
            fn from(($($x),*,): ($($x),*,)) -> Self {
                Self::Tuple(Tuple(vec![$($x.into()),*]))
            }
        }
    }
}

macro_rules! from_struct {
    {
        $($x:ident : $t:ty),*
    } => {
        #[allow(non_camel_case_types)]
        impl From<($($t),*,)> for Value {
            fn from(($($x),*,): ($($t),*,)) -> Self {
                Self::Struct(Struct(vec![$($x),*]))
            }
        }
    }
}

from_primitive!(I8(i8));
from_primitive!(I16(i16));
from_primitive!(I32(i32));
from_primitive!(I64(i64));
from_primitive!(U8(u8));
from_primitive!(U16(u16));
from_primitive!(U32(u32));
from_primitive!(U64(u64));
from_primitive!(Bf16(bf16));
from_primitive!(F16(f16));
from_primitive!(F32(f32));
from_primitive!(F64(f64));
from_primitive!(Char(char));
from_primitive!(Str(String));
from_primitive!(Bool(bool));

from_tuple!(x0);
from_tuple!(x0, x1);
from_tuple!(x0, x1, x2);
from_tuple!(x0, x1, x2, x3);
from_tuple!(x0, x1, x2, x3, x4);
from_tuple!(x0, x1, x2, x3, x4, x5);
from_tuple!(x0, x1, x2, x3, x4, x5, x6);
from_tuple!(x0, x1, x2, x3, x4, x5, x6, x7);

use Field as F;
from_struct!(x0: F);
from_struct!(x0: F, x1: F);
from_struct!(x0: F, x1: F, x2: F);
from_struct!(x0: F, x1: F, x2: F, x3: F);
from_struct!(x0: F, x1: F, x2: F, x3: F, x4: F);
from_struct!(x0: F, x1: F, x2: F, x3: F, x4: F, x5: F);
from_struct!(x0: F, x1: F, x2: F, x3: F, x4: F, x5: F, x6: F);
from_struct!(x0: F, x1: F, x2: F, x3: F, x4: F, x5: F, x6: F, x7: F);

impl From<&'_ str> for Value {
    fn from(s: &'_ str) -> Self {
        Value::Str(s.into())
    }
}

#[derive(Clone, Debug, From)]
pub struct Tuple(Vec<Value>);

#[derive(Clone, Debug, From)]
pub struct Vector(Vec<Value>);

#[derive(Clone, Debug, From)]
pub struct Array(Vec<Value>);

#[derive(Clone, Debug, From)]
pub struct Struct(Vec<Field>);

#[derive(Clone, Debug)]
pub struct Variant(String, Box<Value>);

#[derive(Debug, Clone)]
pub struct Field {
    name: String,
    val: Value,
}

impl Field {
    pub fn new(name: impl Into<String>, val: impl Into<Value>) -> Self {
        Self {
            name: name.into(),
            val: val.into(),
        }
    }
}
