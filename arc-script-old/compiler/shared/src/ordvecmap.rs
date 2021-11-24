use crate::VecMap;
use itertools::Itertools;
use shrinkwraprs::Shrinkwrap;

#[derive(Shrinkwrap, Debug, Clone)]
pub struct OrdVecMap<K: Eq, V>(pub VecMap<K, V>);

impl<K: Eq + Ord + Copy, V> From<VecMap<K, V>> for OrdVecMap<K, V> {
    fn from(mut vecmap: VecMap<K, V>) -> Self {
        OrdVecMap(vecmap.drain().sorted_by_key(|(k, _)| *k).collect())
    }
}

impl<K: Eq + Ord + Copy, V> From<Vec<(K, V)>> for OrdVecMap<K, V> {
    fn from(mut vec: Vec<(K, V)>) -> Self {
        OrdVecMap(vec.drain(..).sorted_by_key(|(k, _)| *k).collect())
    }
}
