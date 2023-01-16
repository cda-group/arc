use std::cell::UnsafeCell;

use crate::iteratee::Iteratee;

pub struct Transform<T> {
    inner: UnsafeCell<Option<T>>,
}

impl<T> Transform<T> {
    pub fn new() -> Self {
        Self {
            inner: UnsafeCell::new(None),
        }
    }
    pub fn iteratee(&self) -> TransformIteratee<T> {
        TransformIteratee(unsafe { &mut *self.inner.get() })
    }
    pub fn iterator(&self) -> TransformIterator<T> {
        TransformIterator(unsafe { &mut *self.inner.get() })
    }
}

pub struct TransformIteratee<'i, T>(&'i mut Option<T>);
pub struct TransformIterator<'i, T>(&'i mut Option<T>);

impl<'i, T> Iterator for TransformIterator<'i, T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.take()
    }
}

impl<'i, T> Iteratee for TransformIteratee<'i, T> {
    type Item = T;
    fn feed(&mut self, item: Self::Item) {
        *self.0 = Some(item);
    }
}
