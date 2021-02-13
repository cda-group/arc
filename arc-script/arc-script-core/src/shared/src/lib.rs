pub mod lower;
pub mod format;

pub use format::Format;
pub use lower::Lower;

pub use anyhow::Result;
pub use derive_more::Constructor as New;
pub use derive_more::From;
pub use educe::Educe;
pub use linear_map::Entry;
pub use shrinkwraprs::Shrinkwrap;
pub use cfg_if::cfg_if;

pub type VecMap<K, V> = linear_map::LinearMap<K, V>;
pub type Map<K, V> = indexmap::map::IndexMap<K, V>;
pub type Set<K> = std::collections::HashSet<K>;
