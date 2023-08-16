use std::borrow::Borrow;
use std::collections::HashSet;
use std::hash::Hash;

use serde::Deserialize;
use serde::Serialize;

use crate::cow::Cow;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct Set<T: Eq + Hash>(pub Cow<HashSet<T>>);

impl<T: Eq + Hash> Set<T> {
    pub fn new() -> Self {
        Self(Cow::new(HashSet::new()))
    }

    pub fn insert(mut self, value: T) -> Self
    where
        T: Clone,
    {
        self.0.update(|this| this.insert(value));
        self
    }

    pub fn remove(mut self, value: impl Borrow<T>) -> Self
    where
        T: Clone,
    {
        self.0.update(|this| this.remove(value.borrow()));
        self
    }

    pub fn contains(&self, value: impl Borrow<T>) -> bool
    where
        T: Clone,
    {
        self.0.contains(value.borrow())
    }

    pub fn group(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.0.iter().cloned().collect()
    }
}

impl<T: Eq + Hash> From<HashSet<T>> for Set<T> {
    fn from(set: HashSet<T>) -> Self {
        Self(Cow::new(set))
    }
}
