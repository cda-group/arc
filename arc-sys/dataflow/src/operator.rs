use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::data::Data;

pub trait Operator: Iterator {
    type S;
    fn state(self) -> Self::S;
}
