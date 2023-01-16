use serde::de::DeserializeOwned;
use serde::ser::Serialize;

pub fn ser<T: Serialize>(v: T) -> Result<Vec<u8>, serde_json::Error> {
    serde_json::to_vec(&v)
}

pub fn deser<T: DeserializeOwned>(v: Option<Vec<u8>>) -> Result<Option<T>, serde_json::Error> {
    v.map(|v| serde_json::from_slice(&v)).transpose()
}
