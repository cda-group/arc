use crate::serde::serialise;
use crate::serde::Serde;
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Deref;
use std::ops::DerefMut;

use futures::Future;
use serde::de::DeserializeOwned;
use serde::Serialize;
use sled::Batch;

use crate::data::Data;
use crate::data::Key;
use crate::db::Database;
use crate::utils::serde::ser;

#[derive(Clone)]
pub struct State<K, T> {
    name: String,
    db: Database,
    cache: HashMap<K, T>,
    default: T,
}

impl<K: Key, T: Data> State<K, T> {
    pub fn new(name: impl ToString, db: Database, default: T) -> Self {
        Self {
            name: name.to_string(),
            db: db.into(),
            cache: HashMap::new(),
            default,
        }
    }
    pub fn get(&mut self, key: K) -> &mut T {
        let default = &self.default;
        self.cache.entry(key).or_insert_with(|| default.clone())
    }
    pub fn batch(&self) -> impl Iterator<Item = (&K, &T)> {
        self.cache.iter()
    }
    pub async fn persist(&mut self) {
        let name = &self.name;
        let batch = self.cache.iter().filter_map(|(k, v)| {
            serialise((name, k))
                .and_then(|k| Ok((k, serialise(v)?)))
                .ok()
        });
        match &self.db {
            #[cfg(feature = "tikv")]
            Database::Remote(db) => db.batch_put(batch).await.expect("tikv: failed to insert"),
            Database::Local(db) => {
                for (k, v) in batch {
                    db.insert(k, v).expect("sled: failed to insert");
                }
            }
        }
    }
}
