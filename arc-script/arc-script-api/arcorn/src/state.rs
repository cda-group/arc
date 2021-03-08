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

/// Arc collections
pub type ArcRef<T, B> = LazyValue<T, B>;
pub type ArcVec<T, B> = EagerAppender<T, B>;
pub type ArcMap<K, V, B> = HashTable<K, V, B>;
pub type ArcSet<K, B> = HashTable<K, (), B>;

/// Arc values (elements of collections)
pub trait ArcValue: ArconValue {}
pub trait ArcKey: ArconKey + Hash + Eq {}

impl<T: ArconValue> ArcValue for T {}
impl<T: ArconKey + Hash + Eq> ArcKey for T {}

/// Arc interface to Arcon value-state.
pub trait ArcRefOps<T: ArcValue, B: Backend>: Sized {
    /// Initializes a new value-state.
    fn arc_ref_new(name: &str, handle: Arc<B>, init: T) -> OperatorResult<Self>;

    /// Initializes a default value-state.
    fn arc_ref_default(name: &str, handle: Arc<B>) -> OperatorResult<Self>;

    /// Updates the value of the state.
    fn arc_ref_write(&mut self, v: T) -> OperatorResult<()>;

    /// Returns the value of the state.
    fn arc_ref_read(&mut self) -> OperatorResult<T>;
}

impl<T: ArcValue, B: Backend> ArcRefOps<T, B> for ArcRef<T, B> {
    fn arc_ref_new(name: &str, handle: Arc<B>, init: T) -> OperatorResult<Self> {
        let mut state = ArcRef::new(name, handle);
        state.arc_ref_write(init)?;
        Ok(state)
    }

    fn arc_ref_default(name: &str, handle: Arc<B>) -> OperatorResult<Self> {
        let mut state = ArcRef::new(name, handle);
        state.arc_ref_write(T::default())?;
        Ok(state)
    }

    fn arc_ref_write(&mut self, v: T) -> OperatorResult<()> {
        self.put(v)?;
        Ok(())
    }

    fn arc_ref_read(&mut self) -> OperatorResult<T> {
        Ok(self.get()?.unwrap().to_mut().clone())
    }
}

pub trait ArcVecOps<T: ArcValue, B: Backend>: Sized {
    /// Initializes a new appender state.
    fn arc_vec_new(name: &str, handle: Arc<B>, init: Vec<T>) -> OperatorResult<Self>;

    /// Initializes a default appender state.
    fn arc_vec_default(name: &str, handle: Arc<B>) -> OperatorResult<Self>;

    /// Pushes a value onto the end of the appender-state.
    fn arc_vec_push(&mut self, data: T) -> OperatorResult<()>;

    /// Returns the result of folding the appender state with a folder function, starting with
    /// and initial value.
    fn arc_vec_fold<A>(&mut self, init: A, folder: fn(A, T) -> A) -> OperatorResult<A>;
}

impl<T: ArcValue, B: Backend> ArcVecOps<T, B> for ArcVec<T, B> {
    fn arc_vec_new(name: &str, handle: Arc<B>, init: Vec<T>) -> OperatorResult<Self> {
        let mut state = ArcVec::new(name, handle);
        init.into_iter().try_for_each(|v| state.arc_vec_push(v))?;
        Ok(state)
    }

    /// Initializes a default appender state.
    fn arc_vec_default(name: &str, handle: Arc<B>) -> OperatorResult<Self> {
        let state = ArcVec::new(name, handle);
        Ok(state)
    }

    fn arc_vec_push(&mut self, data: T) -> OperatorResult<()> {
        self.append(data)?;
        Ok(())
    }

    fn arc_vec_fold<A>(&mut self, init: A, folder: fn(A, T) -> A) -> OperatorResult<A> {
        Ok(self.consume()?.into_iter().fold(init, folder))
    }
}

pub trait ArcMapOps<K: ArcKey, V: ArcValue, B: Backend>: Sized {
    /// Initializes a new map-state.
    fn arc_map_new(name: &str, handle: Arc<B>, init: HashMap<K, V>) -> OperatorResult<Self>;

    /// Initializes a default map-state.
    fn arc_map_default(name: &str, handle: Arc<B>) -> OperatorResult<Self>;

    /// Inserts a value into the map-state.
    fn arc_map_insert(&mut self, key: K, val: V) -> OperatorResult<()>;

    /// Returns `true` if the map-state contains the specified key.
    fn arc_map_contains(&self, key: K) -> OperatorResult<bool>;

    /// Returns the value associated to the given key in the map-state, panics if it does not
    /// exist.
    fn arc_map_get_unchecked(&mut self, key: K) -> OperatorResult<V>;

    /// Removes the specified key and its associated value from the map-state.
    fn arc_map_del(&mut self, key: K) -> OperatorResult<()>;
}

/// Arc Map abstraction
impl<K: ArcKey, V: ArcValue, B: Backend> ArcMapOps<K, V, B> for ArcMap<K, V, B> {
    fn arc_map_new(name: &str, handle: Arc<B>, init: HashMap<K, V>) -> OperatorResult<Self> {
        let mut state = ArcMap::new(name, handle);
        init.into_iter().try_for_each(|(k, v)| state.put(k, v))?;
        Ok(state)
    }

    fn arc_map_default(name: &str, handle: Arc<B>) -> OperatorResult<Self> {
        let state = ArcMap::new(name, handle);
        Ok(state)
    }

    fn arc_map_insert(&mut self, key: K, val: V) -> OperatorResult<()> {
        self.put(key, val)?;
        Ok(())
    }

    fn arc_map_contains(&self, key: K) -> OperatorResult<bool> {
        Ok(self.get(&key)?.is_some())
    }

    fn arc_map_get_unchecked(&mut self, key: K) -> OperatorResult<V> {
        Ok(self.get(&key)?.unwrap().clone())
    }

    fn arc_map_del(&mut self, key: K) -> OperatorResult<()> {
        self.remove(&key)?;
        Ok(())
    }
}

/// Arc Set abstraction
pub trait ArcSetOps<K: ArcKey, B: Backend>: Sized {
    /// Returns a new set, initialized with the specified ephemeral set.
    fn arc_set_new(name: &str, handle: Arc<B>, init: HashSet<K>) -> OperatorResult<Self>;

    /// Returns an empty set.
    fn arc_set_default(name: &str, handle: Arc<B>) -> OperatorResult<Self>;

    /// Inserts a key into the set.
    fn arc_set_add(&mut self, key: K) -> OperatorResult<()>;

    /// Returns `true` if the set contains the specified key, else `false`.
    fn arc_set_contains(&self, key: K) -> OperatorResult<bool>;

    /// Removes the specified key from the set.
    fn arc_set_del(&mut self, key: K) -> OperatorResult<()>;
}

impl<K: ArcKey, B: Backend> ArcSetOps<K, B> for ArcSet<K, B> {
    fn arc_set_new(name: &str, handle: Arc<B>, init: HashSet<K>) -> OperatorResult<Self> {
        let mut state = ArcSet::new(name, handle);
        init.into_iter().try_for_each(|k| state.put(k, ()))?;
        Ok(state)
    }

    fn arc_set_default(name: &str, handle: Arc<B>) -> OperatorResult<Self> {
        let state = ArcSet::new(name, handle);
        Ok(state)
    }

    fn arc_set_add(&mut self, key: K) -> OperatorResult<()> {
        self.put(key, ())?;
        Ok(())
    }

    fn arc_set_contains(&self, key: K) -> OperatorResult<bool> {
        Ok(self.get(&key)?.is_some())
    }

    fn arc_set_del(&mut self, key: K) -> OperatorResult<()> {
        self.remove(&key)?;
        Ok(())
    }
}
