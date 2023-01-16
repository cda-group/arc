use crate::prelude::*;
use crate::serde::Serde;

use macros::export;
use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde::Serializer;

use std::cell::UnsafeCell;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Deref;
use std::ops::DerefMut;

#[derive(Debug, Unpin)]
pub struct Cell<T: Data>(pub UnsafeCell<T>);

impl<T: Data + Hash> Hash for Cell<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.get_ref().hash(state)
    }
}

impl<T: Data + Eq> Eq for Cell<T> {}

impl<T: Data + PartialEq> PartialEq for Cell<T> {
    fn eq(&self, other: &Self) -> bool {
        self.get_ref().eq(&other.get_ref())
    }
}

impl<T: Data> Clone for Cell<T> {
    fn clone(&self) -> Self {
        unsafe { Self(UnsafeCell::new((*self.0.get()).clone())) }
    }
}

impl<T: Data> Serialize for Cell<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.get_ref().serialize(serializer)
    }
}

impl<'de, T: Data> Deserialize<'de> for Cell<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Cell(UnsafeCell::new(T::deserialize(deserializer)?)))
    }
}

impl<T: Data> Deref for Cell<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.0.get() }
    }
}

impl<T: Data> DerefMut for Cell<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.0.get() }
    }
}

#[export]
impl<T: Data> Cell<T> {
    pub fn new(v: T) -> Cell<T> {
        Cell(UnsafeCell::new(v))
    }
    pub fn get(self) -> T {
        unsafe { (*self.0.get()).clone() }
    }
    pub fn set(self, v: T) {
        unsafe { *self.0.get() = v }
    }
}

impl<T: Data> Cell<T> {
    pub fn get_ref(&self) -> &T {
        unsafe { &*self.0.get() }
    }
    pub fn get_mut_ref(&self) -> &mut T {
        unsafe { &mut *self.0.get() }
    }
}
