use macros::DeepClone;
use macros::Send;
use macros::Sync;
use macros::Unpin;
use serde::Deserialize;
use serde::Serialize;

use crate::traits::DeepClone;

#[derive(
    DeepClone,
    Clone,
    Debug,
    Send,
    Sync,
    Unpin,
    Serialize,
    Deserialize,
    Eq,
    PartialEq,
    Hash,
    Ord,
    PartialOrd,
)]
#[repr(C)]
pub struct Cow<T>(pub std::rc::Rc<T>);

impl<T> std::ops::Deref for Cow<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl<T> Cow<T> {
    /// Create a new cow.
    pub fn new(value: T) -> Cow<T> {
        Cow(std::rc::Rc::new(value))
    }

    /// Copy the value of the cow.
    pub fn copy(&self) -> T
    where
        T: Clone,
    {
        self.0.as_ref().clone()
    }

    /// Take the value of the cow. If the cow is shared, it will be cloned.
    pub fn take(self) -> T
    where
        T: Clone,
    {
        match std::rc::Rc::try_unwrap(self.0) {
            Ok(this) => this,
            Err(this) => this.as_ref().clone(),
        }
    }

    /// Update the value of the cow. If the cow is shared, it will be cloned.
    pub fn update<O>(&mut self, f: impl FnOnce(&mut T) -> O) -> O
    where
        T: Clone,
    {
        if let Some(this) = std::rc::Rc::get_mut(&mut self.0) {
            f(this)
        } else {
            let mut this = self.copy();
            let o = f(&mut this);
            self.0 = std::rc::Rc::new(this);
            o
        }
    }

    /// Map the value of the cow. If the cow is shared, it will be cloned.
    pub fn map<O>(self, f: impl FnOnce(T) -> O) -> O
    where
        T: Clone,
    {
        match std::rc::Rc::try_unwrap(self.0) {
            Ok(this) => f(this),
            Err(this) => f(this.as_ref().clone()),
        }
    }

    /// Set the value of the cow. If the cow is shared, it will be cloned.
    /// This is useful to prevent reallocating new cells.
    pub fn set(&mut self, value: T) {
        match std::rc::Rc::get_mut(&mut self.0) {
            Some(this) => *this = value,
            None => self.0 = std::rc::Rc::new(value),
        }
    }
}
