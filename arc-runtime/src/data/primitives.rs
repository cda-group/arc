use crate::context::Context;
use crate::data::convert_reflexive;
use crate::data::garbage::alloc_identity;
use crate::data::strings::String;
use crate::data::Alloc;
use crate::data::DynSendable;
use crate::data::DynSharable;

pub use bool;
pub use char;
pub use f32;
pub use f64;
pub use i128;
pub use i16;
pub use i32;
pub use i64;
pub use i8;
pub use u128;
pub use u16;
pub use u32;
pub use u64;
pub use u8;
#[allow(non_camel_case_types)]
pub type unit = ();
pub type Unit = ();
pub use std::ops::Range;

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
// convert_reflexive!(char);
convert_reflexive!(unit);

alloc_identity!(i8);
alloc_identity!(i16);
alloc_identity!(i32);
alloc_identity!(i64);
alloc_identity!(i128);
alloc_identity!(u8);
alloc_identity!(u16);
alloc_identity!(u32);
alloc_identity!(u64);
alloc_identity!(f32);
alloc_identity!(f64);
alloc_identity!(bool);
alloc_identity!(char);
alloc_identity!(unit);

#[allow(non_upper_case_globals)]
pub const unit: unit = ();
#[allow(non_upper_case_globals)]
pub const Unit: unit = ();

pub fn assert(b: bool, _ctx: Context) {
    assert!(b);
}

pub fn panic(s: String, _ctx: Context) {
    panic!("{}", s.as_str())
}

pub fn print(s: String, _ctx: Context) {
    println!("{}", s.as_str())
}
