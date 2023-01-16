use std::collections::HashMap;
use std::rc::Rc;

use derive_more::From;
use macros::export;
use macros::Unpin;
use serde::Deserialize;
use serde::Serialize;

use super::cell::Cell;
use super::Data;
use super::Key;

#[derive(Clone, From, Debug, Unpin)]
pub struct Dict<K: Key, V: Data>(pub Rc<HashMap<K, V>>);

impl<K: Key, V: Data> Dict<K, V> {
    fn get_mut(&mut self) -> &mut HashMap<K, V> {
        unsafe { Rc::get_mut_unchecked(&mut self.0) }
    }
}

#[export]
impl<K: Key, V: Data> Dict<K, V> {
    pub fn new() -> Dict<K, V> {
        Dict(Rc::new(HashMap::new()))
    }
    pub fn get(mut self, key: K) -> Option<V> {
        self.get_mut().get(&key).cloned()
    }
    pub fn insert(mut self, key: K, val: V) {
        self.get_mut().insert(key, val);
    }
}
