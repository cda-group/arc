use std::rc::Rc;

use crate::prelude::*;

use derive_more::Deref;
use derive_more::DerefMut;
use derive_more::From;
use macros::export;
use serde::Deserialize;
use serde::Serialize;

#[derive(Clone, Hash, Eq, PartialEq, Deref, DerefMut, Debug, Unpin, Serialize, Deserialize)]
pub struct Str(pub Rc<String>);

impl Str {
    fn get_mut(&mut self) -> &mut String {
        unsafe { Rc::get_mut_unchecked(&mut self.0) }
    }
}

#[export]
impl Str {
    pub fn new() -> Str {
        Str(Rc::new(std::string::String::new()))
    }

    pub fn with_capacity(capacity: usize) -> Str {
        Str(Rc::new(std::string::String::with_capacity(capacity)))
    }

    pub fn push_char(mut self, ch: char) {
        self.get_mut().push(ch)
    }

    pub fn push_str(mut self, s: &str) {
        self.get_mut().push_str(s)
    }

    pub fn from_str(s: &str) -> Str {
        let new = Str::with_capacity(s.len());
        new.clone().push_str(s);
        new
    }

    pub fn remove(mut self, idx: u32) -> char {
        self.get_mut().remove(idx as usize)
    }

    pub fn insert_char(mut self, idx: u32, ch: char) {
        self.get_mut().insert(idx as usize, ch)
    }

    pub fn is_empty(self) -> bool {
        self.0.is_empty()
    }

    pub fn split_off(mut self, at: u32) -> Str {
        Str(Rc::new(self.get_mut().split_off(at as usize)))
    }

    pub fn clear(mut self) {
        self.get_mut().clear()
    }

    pub fn len(self) -> u32 {
        self.0.len() as u32
    }

    pub fn from_i32(i: i32) -> Str {
        let mut new = Str::new();
        new.get_mut().push_str(&i.to_string());
        new
    }

    pub fn eq(self, other: Str) -> bool {
        self.0.eq(&other.0)
    }

    pub fn concat(self, other: Str) -> Str {
        let mut new = Str::new();
        new.get_mut().push_str(&self.0);
        new.get_mut().push_str(&other.0);
        new
    }
}
