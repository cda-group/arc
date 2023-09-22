use macros::Send;
use macros::Sync;
use macros::Unpin;
use serde::Deserialize;
use serde::Serialize;

use crate::cow::Cow;
use crate::iterator::Iter;
use crate::option::Option;
use crate::traits::DeepClone;

#[derive(Clone, Debug, Send, Sync, Unpin, Serialize, Deserialize, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[repr(C)]
pub struct Vec<T>(pub Cow<std::vec::Vec<T>>);

impl<T: DeepClone> DeepClone for Vec<T> {
    fn deep_clone(&self) -> Self {
        Self(self.0.deep_clone())
    }
}

impl<T> std::ops::Index<usize> for Vec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T> Vec<T> {
    /// Returns an iterator over the slice.
    pub fn iter(self) -> Iter<T, impl Iterator<Item = T> + Clone>
    where
        T: Clone,
    {
        Iter::new(self.0.take().into_iter())
    }

    pub fn new() -> Vec<T> {
        Vec(Cow::new(std::vec::Vec::<T>::new()))
    }

    pub fn len(self) -> usize {
        self.0.len()
    }

    pub fn push(mut self, value: T) -> Self
    where
        T: Clone,
    {
        self.0.update(|this| this.push(value));
        self
    }

    pub fn pop(mut self) -> (Self, Option<T>)
    where
        T: Clone,
    {
        let x = self.0.update(|this| this.pop());
        (self, Option::from(x))
    }

    pub fn remove(mut self, index: usize) -> (Self, T)
    where
        T: Clone,
    {
        let x = self.0.update(|this| this.remove(index));
        (self, x)
    }

    pub fn get(self, index: usize) -> Option<T>
    where
        T: Clone,
    {
        Option::from(self.0.get(index).map(|x| x.clone()))
    }

    pub fn insert(mut self, index: usize, value: T) -> Self
    where
        T: Clone,
    {
        self.0.update(|this| this.insert(index, value));
        self
    }

    pub fn is_empty(self) -> bool {
        self.0.is_empty()
    }

    pub fn sort(mut self) -> Self
    where
        T: Clone + PartialOrd,
    {
        self.0
            .update(|this| this.sort_by(|a, b| a.partial_cmp(b).unwrap()));
        self
    }

    pub fn truncate(mut self, len: usize) -> Self
    where
        T: Clone,
    {
        self.0.update(|this| this.truncate(len));
        self
    }
}

impl<T> From<std::vec::Vec<T>> for Vec<T> {
    fn from(vec: std::vec::Vec<T>) -> Self {
        Vec(Cow::new(vec))
    }
}
