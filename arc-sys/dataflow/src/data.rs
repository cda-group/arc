pub mod cell;
pub mod dataframe;
pub mod dict;
pub mod primitive;
pub mod series;
pub mod set;
pub mod string;
pub mod vector;

use std::fmt::Debug;
use std::hash::Hash;

use crate::serde::Serde;

pub trait Data: Clone + Serde + Unpin + Debug + 'static {}
impl<T> Data for T where T: Clone + Serde + Unpin + Debug + 'static {}

pub trait Key: Data + Hash + Eq {}
impl<T> Key for T where T: Data + Hash + Eq {}
