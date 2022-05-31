use crate::context::Context;
// use crate::data::strings::String;
use crate::data::gc::Heap;
use crate::data::strings::Str;
use crate::data::Data;
use macros::rewrite;

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

#[allow(non_upper_case_globals)]
pub const unit: unit = ();
#[allow(non_upper_case_globals)]
pub const Unit: unit = ();

use crate::prelude::*;

#[rewrite(generic)]
pub fn bool_assert(b: bool) {
    assert!(b);
}

#[rewrite(generic)]
#[allow(non_snake_case)]
pub fn Str_panic(s: Str) {
    panic!("{}", s.as_str())
}

#[rewrite(generic)]
#[allow(non_snake_case)]
pub fn Str_print(s: Str) {
    println!("{}", s.as_str())
}
