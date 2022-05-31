pub mod cells;
pub mod channels;
#[cfg(feature = "dataframes")]
pub mod dataframe;
pub mod unsafe_channels;
pub mod atomic_channel;
// pub mod erased;
pub mod dicts;
pub mod functions;
pub mod gc;
pub mod primitives;
pub mod serde;
pub mod series;
pub mod strings;
pub mod vectors;

// pub mod deserialiser;
// pub mod error;
// pub mod serialiser;

use std::collections::HashMap;
use std::collections::HashSet;

// use crate::data::serde::DynSerde;
use crate::data::serde::Serde;

use dyn_clone::DynClone;

use std::fmt::Debug;
use std::hash::Hash;

use crate::prelude::*;

// pub trait DynData: Send + Sync + Unpin + DynClone + Trace + Debug + DynSerde {}
// impl<T> DynData for T where T: Send + Sync + Unpin + DynClone + Trace + Debug + DynSerde {}

pub trait Data: Sized + Send + Sync + Unpin + Trace + Debug + Serde + Debug + Clone + Copy {}
impl<T> Data for T where
    T: Sized + Send + Sync + Unpin + Trace + Debug + Serde + Debug + Clone + Copy
{
}

pub trait Key: Data + Eq + Hash {}
impl<T> Key for T where T: Data + Eq + Hash {}

// dyn_clone::clone_trait_object!(DynData);
