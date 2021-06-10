pub mod lower;
pub mod format;
pub mod macros;
pub mod arena;

pub use arena::Arena;
pub use format::Format;
pub use lower::Lower;

pub use anyhow::Result;
pub use boolinator::Boolinator as Bool;
pub use cfg_if::cfg_if;
pub use derive_more::Constructor as New;
pub use derive_more::From;
pub use derive_more::Into;
pub use educe::Educe;
pub use fxhash::FxBuildHasher as Hasher;
pub use linear_map::Entry;
pub use shrinkwraprs::Shrinkwrap;

pub use indexmap::map::Entry as OrdMapEntry;
pub use linear_map::Entry as VecMapEntry;
pub use std::collections::hash_map::Entry as MapEntry;

pub type VecMap<K, V> = linear_map::LinearMap<K, V>;
pub type VecDeque<V> = std::collections::VecDeque<V>;
pub type OrdMap<K, V> = indexmap::map::IndexMap<K, V, Hasher>;
pub type Map<K, V> = std::collections::HashMap<K, V, Hasher>;
pub type Set<K> = std::collections::HashSet<K, Hasher>;

pub use itertools;
