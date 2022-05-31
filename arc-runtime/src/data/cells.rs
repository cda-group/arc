use crate::prelude::*;

use std::cell::UnsafeCell;

pub struct Cell<T: Data>(pub UnsafeCell<T>);

#[rewrite]
impl<T: Data> Cell<T> {
    pub fn new(v: T, ctx: Context<impl Execute>) -> Cell<T> {
        Cell(UnsafeCell::new(v))
    }
    pub fn get(&self, ctx: Context<impl Execute>) -> T {
        unsafe { *self.0.get() }
    }
    pub fn set(&self, v: T, ctx: Context<impl Execute>) {
        unsafe { *self.0.get() = v }
    }
}
