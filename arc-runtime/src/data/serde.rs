use crate::context::Context;
use crate::data::gc::Heap;
use crate::data::Data;
use crate::dispatch::Execute;
use serde_state::DeserializeState;
use serde_state::SerializeState;
use serde_traitobject::Deserialize as DynDeserialize;
use serde_traitobject::Serialize as DynSerialize;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use crate::prelude::Send;
use crate::prelude::Sync;

pub trait Serialize: SerializeState<SerdeState> {}
pub trait Deserialize: for<'de> DeserializeState<'de, SerdeState> {}
pub trait Serde: Serialize + Deserialize {}
// pub trait DynSerde: DynSerialize + DynDeserialize {}

impl<T> Serialize for T where T: SerializeState<SerdeState> {}
impl<T> Deserialize for T where T: for<'de> DeserializeState<'de, SerdeState> {}
impl<T> Serde for T where T: Serialize + Deserialize {}
// impl<T> DynSerde for T where T: DynSerialize + DynDeserialize {}

pub fn serialise<T: Data>(data: T, serde: SerdeState) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut s = serde_json::Serializer::new(&mut buf);
    T::serialize_state(&data, &mut s, &serde).unwrap();
    buf
}

pub fn deserialise<T: Data>(data: Vec<u8>, mut serde: SerdeState) -> T {
    let mut d = serde_json::Deserializer::from_slice(&data);
    T::deserialize_state(&mut serde, &mut d).unwrap()
}

#[derive(Copy, Clone, Send, Sync)]
pub struct SerdeState(*mut Core);

impl std::ops::Deref for SerdeState {
    type Target = Core;
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.0 }
    }
}

impl std::ops::DerefMut for SerdeState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.0 }
    }
}

impl SerdeState {
    pub fn new(heap: Heap) -> Self {
        Self(Box::into_raw(Box::new(Core::new(heap))))
    }
}

pub struct Core {
    pub serialized: UnsafeCell<HashSet<usize>>,
    pub deserialized: HashMap<usize, usize>,
    pub heap: Heap,
}

impl Core {
    pub fn new(heap: Heap) -> Self {
        Self {
            serialized: UnsafeCell::new(HashSet::new()),
            deserialized: HashMap::new(),
            heap,
        }
    }

    #[allow(clippy::mut_from_ref)]
    pub fn serialized(&self) -> &mut HashSet<usize> {
        unsafe { &mut *self.serialized.get() }
    }
}
