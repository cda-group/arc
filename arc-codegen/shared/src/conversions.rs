use crate::primitives::*;

pub trait Convert {
    type T;
    fn convert(self) -> Self::T;
}

macro_rules! convert_reflexive {
    { $ty:ty } => {
        impl Convert for $ty {
            type T = $ty;
            fn convert(self) -> Self::T {
                self
            }
        }
    }
}

convert_reflexive!(i8);
convert_reflexive!(i16);
convert_reflexive!(i32);
convert_reflexive!(i64);
convert_reflexive!(i128);

convert_reflexive!(u8);
convert_reflexive!(u16);
convert_reflexive!(u32);
convert_reflexive!(u64);
convert_reflexive!(u128);

convert_reflexive!(f32);
convert_reflexive!(f64);

convert_reflexive!(bool);
convert_reflexive!(char);

convert_reflexive!(Unit);
