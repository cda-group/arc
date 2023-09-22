#![allow(unused)]

use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Index;
use std::ops::IndexMut;
use std::path::PathBuf;

use serde::de::DeserializeOwned;
use serde::Serialize;

use builtins::traits::Data;
use builtins::traits::Key;

#[derive(Clone)]
pub struct Database(
    // tikv_client::RawClient,
    sled::Db,
    // HashMap<String, String>,
);

impl Database {
    // #[cfg(feature = "tikv")]
    // pub fn new(addr: &str) -> Self {
    //     let db = tokio::runtime::Runtime::new()
    //         .expect("failed to create runtime")
    //         .block_on(tikv_client::RawClient::new(vec![addr]).await)
    //         .expect("Failed to connect to tikv");
    //     Self(db)
    // }

    // #[cfg(feature = "local")]
    pub fn new(path: impl AsRef<std::path::Path>) -> Self {
        let db = sled::open(path).expect("Failed to connect to sled");
        Self(db)
    }

    // #[cfg(not(any(feature = "tikv", feature = "sled")))]
    // pub fn new(path: &str) -> Self {
    //     let db = sled::open(path).expect("Failed to connect to sled");
    //     Self(db)
    // }
}

#[derive(Clone)]
pub struct State<K, T> {
    name: String,
    db: Database,
    uncommitted: HashMap<K, T>,
    default: T,
}

impl<K: Data, T: Data> State<K, T> {
    pub fn new(name: impl ToString, db: Database, default: T) -> Self {
        Self {
            name: name.to_string(),
            db,
            uncommitted: HashMap::new(),
            default,
        }
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut T)> {
        self.uncommitted.iter_mut()
    }
    // #[cfg(feature = "remote")]
    // pub async fn snapshot(&mut self, snapshot_version: usize) {
    //     let name = &self.name;
    //     let batch = self.uncommitted.iter().filter_map(|(key, data)| {
    //         let key = ser((name, key)).ok()?;
    //         let data = ser(data).ok()?;
    //         Some((key, data))
    //     });
    //     db.batch_put(batch).await.expect("tikv: failed to insert")
    // }
    pub async fn snapshot(&mut self, snapshot_version: usize) {
        self.uncommitted
            .iter()
            .filter_map(|(key, data)| {
                let key = ser((key, snapshot_version)).ok()?;
                let data = ser(data).ok()?;
                Some((key, data))
            })
            .for_each(|(key, data)| {
                self.db.0.insert(key, data).expect("sled: failed to insert");
            });
    }
}

impl<K: Key, T: Data> IndexMut<K> for State<K, T> {
    fn index_mut(&mut self, key: K) -> &mut T {
        let default = &self.default;
        self.uncommitted
            .entry(key)
            .or_insert_with(|| default.clone())
    }
}

impl<K: Key, T: Data> Index<K> for State<K, T> {
    type Output = T;
    fn index(&self, key: K) -> &T {
        self.uncommitted.get(&key).unwrap_or(&self.default)
    }
}

pub fn ser<T: Serialize>(v: T) -> Result<Vec<u8>, serde_json::Error> {
    serde_json::to_vec(&v)
}

pub fn deser<T: DeserializeOwned>(v: Option<Vec<u8>>) -> Result<Option<T>, serde_json::Error> {
    v.map(|v| serde_json::from_slice(&v)).transpose()
}
