pub(crate) type VecMap<K, V> = linear_map::LinearMap<K, V>;
pub(crate) use linear_map::Entry;
pub(crate) type Map<K, V> = indexmap::map::IndexMap<K, V>;
pub(crate) type Set<K> = std::collections::HashSet<K>;
pub use derive_more::Constructor as New;
pub use derive_more::From;

/// Trait for lowering `Self` into `T`.
pub(crate) trait Lower<T, C> {
    fn lower(&self, ctx: &mut C) -> T;
}

pub(crate) mod display;
