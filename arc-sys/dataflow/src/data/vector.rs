use std::rc::Rc;

use derive_more::From;
use macros::export;
use macros::Send;
use macros::Sync;
use macros::Trace;
use macros::Unpin;
use serde::Deserialize;
use serde::Serialize;

use super::cell::Cell;
use super::Data;

#[derive(Clone, Debug, Send, Sync, Unpin, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Vector<T: Data>(pub Rc<Vec<T>>);

impl<T: Data> Vector<T> {
    fn get_mut(&mut self) -> &mut Vec<T> {
        unsafe { Rc::get_mut_unchecked(&mut self.0) }
    }

    pub fn retain<F>(mut self, f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.get_mut().retain(f);
    }
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        self.get_mut().as_mut_slice()
    }

    pub fn dedup(mut self)
    where
        T: PartialEq,
    {
        self.get_mut().dedup();
    }

    pub fn shrink_to(mut self, min_capacity: usize) {
        self.get_mut().shrink_to(min_capacity);
    }

    pub fn resize(mut self, new_len: usize, value: T) {
        self.get_mut().resize(new_len, value);
    }
}

#[export]
impl<T: Data> Vector<T> {
    pub fn iterator(self) -> VectorIterator<T> {
        VectorIterator::new(self)
    }

    pub fn new() -> Vector<T> {
        Vector(Rc::new(Vec::<T>::new()))
    }

    pub fn from_vec(vec: Vec<T>) -> Vector<T> {
        Vector(Rc::new(vec))
    }

    pub fn with_capacity(capacity: usize) -> Vector<T> {
        Vector(Rc::new(Vec::<T>::with_capacity(capacity)))
    }

    pub fn capacity(self) -> usize {
        self.0.capacity()
    }

    pub fn len(self) -> usize {
        self.0.len()
    }

    pub fn clear(mut self) {
        self.get_mut().clear();
    }

    pub fn push(mut self, value: T) {
        self.get_mut().push(value);
    }

    pub fn pop(mut self) -> Option<T> {
        self.get_mut().pop()
    }

    pub fn remove(mut self, index: usize) -> T {
        self.get_mut().remove(index)
    }

    pub fn get(self, index: usize) -> T {
        self.0[index].clone()
    }

    pub fn insert(mut self, index: usize, value: T) {
        self.get_mut().insert(index, value);
    }

    pub fn is_empty(self) -> bool {
        self.0.is_empty()
    }
}

#[allow(non_snake_case)]
pub fn Vector_dedup<T: Data>(self_param: Vector<T>)
where
    T: PartialEq,
{
    Vector::dedup(self_param)
}

#[derive(Clone, From, Debug, Send, Sync, Unpin, Serialize, Deserialize)]
#[serde(bound = "")]
#[from(forward)]
pub struct VectorIterator<T: Data> {
    vec: Vector<T>,
    offset: u64,
}

#[export]
impl<T: Data> VectorIterator<T> {
    fn new(vec: Vector<T>) -> VectorIterator<T> {
        VectorIterator { vec, offset: 0 }
    }

    pub fn is_empty(self) -> bool {
        self.offset >= self.vec.len() as u64
    }

    pub fn next(mut self) -> T {
        let index = self.offset as usize;
        self.offset += 1;
        self.vec.get(index)
    }
}
