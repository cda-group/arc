use serde::Deserialize;
use serde::Serialize;

pub mod csv {
    pub mod de;
    pub mod ser;
}

pub mod json {
    pub mod de;
    pub mod ser;
}

pub trait Decode {
    type Error: std::error::Error;
    fn decode<'de, T>(&mut self, input: &'de [u8]) -> Result<T, Self::Error>
    where
        T: Deserialize<'de>;
}

pub trait Encode {
    type Error: std::error::Error;
    fn encode<T>(&mut self, input: &T, output: &mut Vec<u8>) -> Result<usize, Self::Error>
    where
        T: Serialize + ?Sized;
    fn content_type(&self) -> &'static str;
}
