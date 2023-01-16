use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::collections::HashSet;

use serde::Deserialize;
use serde::Serialize;

pub trait Serde: Serialize + for<'a> Deserialize<'a> {}
impl<T> Serde for T where T: Serialize + for<'a> Deserialize<'a> {}

pub fn serialise<T: Serialize>(data: T) -> Result<Vec<u8>, serde_json::Error> {
    let mut buffer = Vec::new();
    let mut serializer = serde_json::Serializer::new(&mut buffer);
    data.serialize(&mut serializer);
    Ok(buffer)
}

pub fn deserialise<'de, T: Deserialize<'de>>(bytes: &'de [u8]) -> Result<T, serde_json::Error> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    T::deserialize(&mut deserializer)
}
