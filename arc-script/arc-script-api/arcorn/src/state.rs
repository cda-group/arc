use arcon::prelude::state;
use arcon::prelude::state::data::Key as ArconKey;
use arcon::prelude::state::data::Value as ArconValue;
use arcon::prelude::state::ArconState;
use arcon::prelude::state::Backend;
use arcon::prelude::OperatorResult;

use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;
use std::sync::Arc;

/// Data abstraction over Arcon value-state.
#[derive(ArconState)]
pub struct Value<T: ArconValue, B: Backend> {
    data: state::Value<T, B>,
}

impl<T: ArconValue, B: Backend> Value<T, B> {
    /// Initializes a new value-state.
    #[inline(always)]
    pub fn new(name: &str, handle: Arc<B>, init: T) -> OperatorResult<Self> {
        let mut state = Self {
            data: state::Value::new(name, handle),
        };
        state.write(init)?;
        Ok(state)
    }

    /// Updates the value of the state.
    #[inline(always)]
    pub fn write(&mut self, v: T) -> OperatorResult<()> {
        self.data.put(v);
        Ok(())
    }

    /// Returns the value of the state.
    #[inline(always)]
    pub fn read(&mut self) -> OperatorResult<T> {
        Ok(self.data.get().unwrap().clone())
    }
}

/// Data abstraction over Arcon appender-state.
#[derive(ArconState)]
pub struct Appender<T: ArconValue, B: Backend> {
    data: state::Appender<T, B>,
}

impl<T: ArconValue, B: Backend> Appender<T, B> {
    /// Initializes a new appender state.
    #[inline(always)]
    pub fn new(name: &str, handle: Arc<B>, init: Vec<T>) -> OperatorResult<Self> {
        let mut state = Self {
            data: state::Appender::new(name, handle),
        };
        init.into_iter().try_for_each(|v| state.push(v))?;
        Ok(state)
    }

    /// Pushes a value onto the end of the appender-state.
    #[inline(always)]
    pub fn push(&mut self, data: T) -> OperatorResult<()> {
        self.data.append(data)
    }

    /// Returns the result of folding the appender state with a folder function, starting with
    /// and initial value.
    #[inline(always)]
    pub fn fold<A>(&mut self, init: A, folder: fn(A, T) -> A) -> OperatorResult<A> {
        Ok(self.data.consume()?.into_iter().fold(init, folder))
    }
}

/// Data abstraction over Arcon map-state.
#[derive(ArconState)]
pub struct Map<K: Eq + Hash + ArconKey, V: ArconValue, B: Backend> {
    data: state::HashTable<K, V, B>,
}

impl<K: Hash + Eq + ArconKey, V: ArconValue, B: Backend> Map<K, V, B> {
    /// Initializes a new map-state.
    #[inline(always)]
    pub fn new(name: &str, handle: Arc<B>, init: HashMap<K, V>) -> OperatorResult<Self> {
        let mut state = Self {
            data: state::HashTable::new(name, handle),
        };
        init.into_iter().try_for_each(|(k, v)| state.insert(k, v))?;
        Ok(state)
    }

    /// Inserts a value into the map-state.
    #[inline(always)]
    pub fn insert(&mut self, key: K, val: V) -> OperatorResult<()> {
        self.data.put(key, val)
    }

    /// Returns `true` if the map-state contains the specified key.
    #[inline(always)]
    pub fn contains(&self, key: K) -> OperatorResult<bool> {
        Ok(self.data.get(&key)?.is_some())
    }

    /// Returns the value associated to the given key in the map-state, panics if it does not
    /// exist.
    #[inline(always)]
    pub fn get_unchecked(&mut self, key: K) -> OperatorResult<V> {
        Ok(self.data.get(&key)?.unwrap().clone())
    }

    /// Removes the specified key and its associated value from the map-state.
    #[inline(always)]
    pub fn remove(&mut self, key: K) -> OperatorResult<()> {
        self.data.remove(&key)?;
        Ok(())
    }
}

/// Data abstraction over Arcon map-state.
#[derive(ArconState)]
pub struct Set<K: Eq + Hash + ArconKey, B: Backend> {
    data: state::HashTable<K, (), B>,
}

impl<K: Hash + Eq + ArconKey, B: Backend> Set<K, B> {
    /// Returns a new set, initialized with the specified ephemeral set.
    #[inline(always)]
    pub fn new(name: &str, handle: Arc<B>, init: HashSet<K>) -> OperatorResult<Self> {
        let mut state = Self {
            data: state::HashTable::new(name, handle),
        };
        init.into_iter().try_for_each(|k| state.insert(k))?;
        Ok(state)
    }

    /// Inserts a key into the set.
    #[inline(always)]
    pub fn insert(&mut self, key: K) -> OperatorResult<()> {
        self.data.put(key, ())
    }

    /// Returns `true` if the set contains the specified key, else `false`.
    #[inline(always)]
    pub fn contains(&self, key: K) -> OperatorResult<bool> {
        Ok(self.data.get(&key)?.is_some())
    }

    /// Removes the specified key from the set.
    #[inline(always)]
    pub fn remove(&mut self, key: K) -> OperatorResult<()> {
        self.data.remove(&key)?;
        Ok(())
    }
}
