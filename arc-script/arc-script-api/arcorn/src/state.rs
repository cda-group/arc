use arcon::prelude::state::data::Key as ArconKey;
use arcon::prelude::state::data::Value as ArconValue;
use arcon::prelude::state::Backend;
use arcon::prelude::AppenderIndex;
use arcon::prelude::EagerAppender;
use arcon::prelude::HashTable;
use arcon::prelude::LazyValue;
use arcon::prelude::OperatorResult;
use arcon::prelude::ValueIndex;

use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;
use std::sync::Arc;

trait ArcValue: ArconValue {}
trait ArcKey: ArconKey + Hash + Eq {}

impl<T: ArconValue> ArcValue for T {}
impl<T: ArconKey + Hash + Eq> ArcKey for T {}

/// Arc interface to Arcon value-state.
trait ValueOps<T: ArcValue, B: Backend>: Sized {
    /// Initializes a new value-state.
    fn new(name: &str, handle: Arc<B>, init: T) -> OperatorResult<Self>;

    /// Updates the value of the state.
    fn write(&mut self, v: T) -> OperatorResult<()>;

    /// Returns the value of the state.
    fn read(&mut self) -> OperatorResult<T>;
}

impl<T: ArcValue, B: Backend> ValueOps<T, B> for LazyValue<T, B> {
    fn new(name: &str, handle: Arc<B>, init: T) -> OperatorResult<Self> {
        let mut state = LazyValue::new(name, handle);
        state.write(init)?;
        Ok(state)
    }

    fn write(&mut self, v: T) -> OperatorResult<()> {
        self.put(v)?;
        Ok(())
    }

    fn read(&mut self) -> OperatorResult<T> {
        Ok(self.get().unwrap().unwrap().to_mut().clone())
    }
}

trait AppenderOps<T: ArcValue, B: Backend>: Sized {
    /// Initializes a new appender state.
    fn new(name: &str, handle: Arc<B>, init: Vec<T>) -> OperatorResult<Self>;

    /// Pushes a value onto the end of the appender-state.
    fn push(&mut self, data: T) -> OperatorResult<()>;

    /// Returns the result of folding the appender state with a folder function, starting with
    /// and initial value.
    fn fold<A>(&mut self, init: A, folder: fn(A, T) -> A) -> OperatorResult<A>;
}

impl<T: ArcValue, B: Backend> AppenderOps<T, B> for EagerAppender<T, B> {
    fn new(name: &str, handle: Arc<B>, init: Vec<T>) -> OperatorResult<Self> {
        let mut state = EagerAppender::new(name, handle);
        init.into_iter().try_for_each(|v| state.push(v))?;
        Ok(state)
    }

    fn push(&mut self, data: T) -> OperatorResult<()> {
        self.append(data)?;
        Ok(())
    }

    fn fold<A>(&mut self, init: A, folder: fn(A, T) -> A) -> OperatorResult<A> {
        Ok(self.consume()?.into_iter().fold(init, folder))
    }
}

trait MapOps<K: ArcKey, V: ArcValue, B: Backend>: Sized {
    /// Initializes a new map-state.
    fn new(name: &str, handle: Arc<B>, init: HashMap<K, V>) -> OperatorResult<Self>;

    /// Inserts a value into the map-state.
    fn insert(&mut self, key: K, val: V) -> OperatorResult<()>;

    /// Returns `true` if the map-state contains the specified key.
    fn contains(&self, key: K) -> OperatorResult<bool>;

    /// Returns the value associated to the given key in the map-state, panics if it does not
    /// exist.
    fn get_unchecked(&mut self, key: K) -> OperatorResult<V>;

    /// Removes the specified key and its associated value from the map-state.
    fn remove(&mut self, key: K) -> OperatorResult<()>;
}

/// Arc Map abstraction
impl<K: ArcKey, V: ArcValue, B: Backend> MapOps<K, V, B> for HashTable<K, V, B> {
    fn new(name: &str, handle: Arc<B>, init: HashMap<K, V>) -> OperatorResult<Self> {
        let mut state = HashTable::new(name, handle);
        init.into_iter().try_for_each(|(k, v)| state.put(k, v))?;
        Ok(state)
    }

    fn insert(&mut self, key: K, val: V) -> OperatorResult<()> {
        self.put(key, val)?;
        Ok(())
    }

    fn contains(&self, key: K) -> OperatorResult<bool> {
        Ok(self.get(&key)?.is_some())
    }

    fn get_unchecked(&mut self, key: K) -> OperatorResult<V> {
        Ok(self.get(&key)?.unwrap().clone())
    }

    fn remove(&mut self, key: K) -> OperatorResult<()> {
        self.remove(&key)?;
        Ok(())
    }
}

/// Arc Set abstraction
trait SetOps<K: ArcKey, B: Backend>: Sized {
    /// Returns a new set, initialized with the specified ephemeral set.
    fn new(name: &str, handle: Arc<B>, init: HashSet<K>) -> OperatorResult<Self>;

    /// Inserts a key into the set.
    fn insert(&mut self, key: K) -> OperatorResult<()>;

    /// Returns `true` if the set contains the specified key, else `false`.
    fn contains(&self, key: K) -> OperatorResult<bool>;

    /// Removes the specified key from the set.
    fn remove(&mut self, key: K) -> OperatorResult<()>;
}

impl<K: ArcKey, B: Backend> SetOps<K, B> for HashTable<K, (), B> {
    fn new(name: &str, handle: Arc<B>, init: HashSet<K>) -> OperatorResult<Self> {
        let mut state = HashTable::new(name, handle);
        init.into_iter().try_for_each(|k| state.put(k, ()))?;
        Ok(state)
    }

    fn insert(&mut self, key: K) -> OperatorResult<()> {
        self.put(key, ())?;
        Ok(())
    }

    fn contains(&self, key: K) -> OperatorResult<bool> {
        Ok(self.get(&key)?.is_some())
    }

    fn remove(&mut self, key: K) -> OperatorResult<()> {
        self.remove(&key)?;
        Ok(())
    }
}
