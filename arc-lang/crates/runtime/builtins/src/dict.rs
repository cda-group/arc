use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;

use serde::Deserialize;
use serde::Serialize;

use crate::cow::Cow;
use crate::traits::Data;
use crate::traits::DeepClone;
use crate::traits::Key;

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[repr(C)]
pub struct Dict<K: Eq + Hash, V>(pub Cow<HashMap<K, V>>);

impl<K: Key, V: Data> DeepClone for Dict<K, V> {
    fn deep_clone(&self) -> Self {
        todo!()
        // let map = self
        //     .0
        //     .iter()
        //     .map(|(k, v)| (k.deep_clone(), v.deep_clone()))
        //     .collect();
        // Dict(map)
    }
}

impl<K: Eq + Hash, V> Dict<K, V> {
    pub fn new() -> Dict<K, V> {
        Dict(Cow::new(HashMap::new()))
    }

    pub fn get(self, key: impl Borrow<K>) -> Option<V>
    where
        K: Clone,
        V: Clone,
    {
        self.0.get(key.borrow()).cloned()
    }

    pub fn insert(mut self, key: K, val: V) -> Self
    where
        K: Clone,
        V: Clone,
    {
        self.0.update(|this| this.insert(key, val));
        self
    }

    pub fn remove(mut self, key: impl Borrow<K>) -> Self
    where
        K: Clone,
        V: Clone,
    {
        self.0.update(|this| this.remove(key.borrow()));
        self
    }

    pub fn contains_key(self, key: impl Borrow<K>) -> bool {
        self.0.contains_key(key.borrow())
    }
}

impl<K: Eq + Hash, V> From<std::collections::HashMap<K, V>> for Dict<K, V> {
    fn from(map: std::collections::HashMap<K, V>) -> Self {
        Dict(Cow::new(map))
    }
}
