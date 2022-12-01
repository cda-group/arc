pub mod functions;
pub mod garbage;
pub mod primitives;
pub mod strings;
pub mod vectors;
#[cfg(feature = "dataframes")]
pub mod dataframe;
pub mod series;
pub mod channels;
pub mod cells;

use crate::data::garbage::Garbage;

use comet::gc_base::GcBase;
use dyn_clone::DynClone;

use std::fmt::Debug;
use std::hash::Hash;

use crate::prelude::*;

pub trait Concrete {
    type Abstract;
}

pub trait Abstract {
    type Concrete;
}

pub trait DynSendable: AsyncSafe + DynClone {
    type T: Sharable;
    fn into_sharable(&self, ctx: Context) -> Self::T;
}

pub trait DynSharable: AsyncSafe + DynClone + Garbage + Debug {
    type T: Sendable;
    fn into_sendable(&self, ctx: Context) -> Self::T;
}

pub trait AsyncSafe: Send + Sync + Unpin {}
impl<T> AsyncSafe for T where T: Send + Sync + Unpin {}

pub trait Sendable: Sized + DynSendable + Clone + Serialize + DeserializeOwned {}
pub trait Sharable: Sized + DynSharable + Clone {}
pub trait DataItem: Sized + Copy + Debug + AsyncSafe {}

impl<T> Sharable for T where T: Sized + DynSharable + Clone {}
impl<T> Sendable for T where T: Sized + DynSendable + Clone + Serialize + DeserializeOwned {}
impl<T> DataItem for T where T: Sized + Copy + Debug + AsyncSafe {}

dyn_clone::clone_trait_object!(<T> DynSharable<T = T>);
dyn_clone::clone_trait_object!(<T> DynSendable<T = T>);

#[macro_export]
macro_rules! convert_reflexive {
    {$({$($impl_generics:tt)+})* $ty:ty $({$($where_clause:tt)+})*} => {
        impl $(<$($impl_generics)+>)* DynSharable for $ty $($($where_clause)+)* {
            type T = Self;
            fn into_sendable(&self, _: Context) -> Self { self.clone() }
        }
        impl $(<$($impl_generics)+>)* DynSendable for $ty $($($where_clause)+)* {
            type T = Self;
            fn into_sharable(&self, _: Context) -> Self { self.clone() }
        }
    }
}

pub use convert_reflexive;
