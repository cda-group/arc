use crate::prelude::*;
use std::collections::HashMap;
use std::marker::PhantomData;

#[derive(Clone, From, Debug, Send, Sync, Trace, Unpin)]
#[serde_state(generic)]
pub struct Dict<K: Key, V: Data>(pub Gc<HashMap<K, V>>);

impl<K: Trace + Hash + Eq, V: Trace> Trace for HashMap<K, V> {
    fn trace(&self, heap: Heap) {
        self.iter().for_each(|(k, v)| {
            k.trace(heap);
            v.trace(heap);
        });
    }

    fn root(&self, heap: Heap) {
        self.iter().for_each(|(k, v)| {
            k.root(heap);
            v.root(heap);
        });
    }

    fn unroot(&self, heap: Heap) {
        self.iter().for_each(|(k, v)| {
            k.unroot(heap);
            v.unroot(heap);
        });
    }

    fn copy(&self, heap: Heap) -> Self {
        self.iter()
            .map(|(k, v)| (k.copy(heap), v.copy(heap)))
            .collect()
    }
}

#[rewrite]
impl<K: Key, V: Data> Dict<K, V> {
    pub fn new(ctx: Context<impl Execute>) -> Dict<K, V> {
        Dict(ctx.heap.allocate(HashMap::new()))
    }
    pub fn get(self, key: K, ctx: Context<impl Execute>) -> Option<V> {
        self.0.get(&key).copied()
    }
    pub fn insert(mut self, key: K, val: V, ctx: Context<impl Execute>) {
        self.0.insert(key, val);
    }
}
