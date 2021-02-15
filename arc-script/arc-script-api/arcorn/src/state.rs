use arcon::prelude::state::data::Key as ArconKey;
use arcon::prelude::state::data::Value as ArconValue;
use arcon::prelude::state::Appender;
use arcon::prelude::state::Backend;
use arcon::prelude::state::HashTable;
use arcon::prelude::state::Value;
use arcon::prelude::OperatorResult;
use arcon::prelude::state::ArconState;

use std::hash::Hash;
use std::sync::Arc;

/// Data abstraction over Arcon Values.
#[derive(ArconState)]
pub struct ArcValue<T: ArconValue, B: Backend> {
    data: Value<T, B>,
}

impl<T: ArconValue, B: Backend> ArcValue<T, B> {
    /// Always assume an initial value.
    #[inline(always)]
    pub fn new(name: &str, handle: Arc<B>, init: T) -> OperatorResult<Self> {
        let mut state = Self {
            data: Value::new(name, handle),
        };
        state.write(init)?;
        Ok(state)
    }

    #[inline(always)]
    pub fn write(&mut self, v: T) -> OperatorResult<()> {
        self.data.put(v);
        Ok(())
    }

    #[inline(always)]
    pub fn read(&mut self) -> OperatorResult<T> {
        Ok(self.data.get().unwrap().clone())
    }
}

/// Data abstraction over Arcon Appenders.
#[derive(ArconState)]
pub struct ArcAppender<T: ArconValue, B: Backend> {
    data: Appender<T, B>,
}

impl<T: ArconValue, B: Backend> ArcAppender<T, B> {
    #[inline(always)]
    pub fn new(name: &str, handle: Arc<B>) -> OperatorResult<Self> {
        let state = Self {
            data: Appender::new(name, handle),
        };
        Ok(state)
    }

    #[inline(always)]
    pub fn push(&mut self, data: T) -> OperatorResult<()> {
        self.data.append(data)
    }

    #[inline(always)]
    pub fn fold<A>(&mut self, init: A, folder: fn(A, T) -> A) -> OperatorResult<A> {
        Ok(self.data.consume()?.into_iter().fold(init, folder))
    }
}

/// Data abstraction over Arcon Maps.
#[derive(ArconState)]
pub struct ArcMap<K: Eq + Hash + ArconKey, V: ArconValue, B: Backend> {
    data: HashTable<K, V, B>,
}

impl<K: Hash + Eq + ArconKey, V: ArconValue, B: Backend> ArcMap<K, V, B> {
    #[inline(always)]
    pub fn new(name: &str, handle: Arc<B>) -> OperatorResult<Self> {
        let state = Self {
            data: HashTable::new(name, handle),
        };
        Ok(state)
    }

    #[inline(always)]
    pub fn insert(&mut self, key: K, val: V) -> OperatorResult<()> {
        self.data.put(key, val)
    }

    #[inline(always)]
    pub fn contains(&self, key: K) -> OperatorResult<bool> {
        Ok(self.data.get(&key)?.is_some())
    }

    #[inline(always)]
    pub fn get_unchecked(&mut self, key: K) -> OperatorResult<V> {
        Ok(self.data.get(&key)?.unwrap().clone())
    }

    #[inline(always)]
    pub fn remove(&mut self, key: K) -> OperatorResult<()> {
        self.data.remove(&key)?;
        Ok(())
    }
}
