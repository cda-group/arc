use serde::Deserialize;
use serde::Serialize;

use crate::cow::Cow;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct Blob(pub Cow<std::vec::Vec<u8>>);

impl Blob {
    pub fn new(bytes: std::vec::Vec<u8>) -> Self {
        Self(Cow::new(bytes))
    }
}
