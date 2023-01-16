use crate::prelude::*;

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
