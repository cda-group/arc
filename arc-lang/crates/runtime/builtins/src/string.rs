use formats::Decode;
use formats::Encode;
use macros::DeepClone;
use macros::Send;
use macros::Unpin;

use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde::Serialize;

use crate::cow::Cow;
use crate::encoding::Encoding;
use crate::traits::DeepClone;
use crate::vec::Vec;

#[derive(Clone, DeepClone, Send, Hash, Eq, PartialEq, Ord, PartialOrd, Debug, Unpin)]
#[repr(C)]
pub enum String {
    Text(&'static str),
    Heap(Cow<std::string::String>),
}

impl Serialize for String {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            String::Text(s) => s.serialize(serializer),
            String::Heap(s) => s.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for String {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = <std::string::String>::deserialize(deserializer)?;
        Ok(String::from(s))
    }
}

impl std::fmt::Display for String {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, r#""{}""#, self.as_ref())
    }
}

impl DeepClone for std::string::String {
    fn deep_clone(&self) -> Self {
        self.clone()
    }
}

impl AsRef<str> for String {
    fn as_ref(&self) -> &str {
        match self {
            String::Text(s) => s,
            String::Heap(s) => s.as_ref(),
        }
    }
}

impl String {
    pub fn new() -> String {
        String::Text("")
    }

    pub fn with_capacity(capacity: usize) -> String {
        String::from(std::string::String::with_capacity(capacity))
    }

    fn update<O>(&mut self, f: impl FnOnce(&mut std::string::String) -> O) -> O {
        match self {
            String::Text(s) => {
                let mut s = std::string::String::from(*s);
                let o = f(&mut s);
                *self = String::Heap(Cow::new(s));
                o
            }
            String::Heap(s) => s.update(f),
        }
    }

    pub fn push(mut self, ch: char) -> Self {
        self.update(|s| s.push(ch));
        self
    }

    pub fn push_string(mut self, other: String) -> Self {
        self.update(|s| s.push_str(other.as_ref()));
        self
    }

    pub fn remove(mut self, idx: usize) -> (Self, char) {
        let c = self.update(|s| s.remove(idx));
        (self, c)
    }

    pub fn insert(mut self, idx: usize, ch: char) -> Self {
        self.update(|s| s.insert(idx, ch));
        self
    }

    pub fn is_empty(self) -> bool {
        self.as_ref().is_empty()
    }

    pub fn split_off(mut self, at: usize) -> (Self, String) {
        let s = self.update(|s| String::from(s.split_off(at)));
        (self, s)
    }

    pub fn lines(self) -> Vec<String> {
        self.as_ref()
            .lines()
            .map(|s| String::from(s.to_string()))
            .collect::<std::vec::Vec<_>>()
            .into()
    }

    pub fn clear(mut self) -> Self {
        self.update(|s| s.clear());
        self
    }

    pub fn len(self) -> usize {
        self.as_ref().len()
    }

    pub fn decode<T: DeserializeOwned>(self, encoding: Encoding) -> T {
        match encoding {
            Encoding::Csv { sep } => formats::csv::de::Reader::<1024>::new(sep)
                .decode(self.as_ref().as_bytes())
                .unwrap(),
            Encoding::Json => formats::json::de::Reader::new()
                .decode(self.as_ref().as_bytes())
                .unwrap(),
        }
    }

    pub fn encode<T: Serialize>(value: T, encoding: Encoding) -> Self {
        let mut output = std::vec::Vec::new();
        match encoding {
            Encoding::Csv { sep } => formats::csv::ser::Writer::new(sep)
                .encode(&value, &mut output)
                .unwrap(),
            Encoding::Json => formats::json::ser::Writer::new()
                .encode(&value, &mut output)
                .unwrap(),
        };
        String::from(std::string::String::from_utf8(output).unwrap())
    }
}

impl<'a> From<&'a str> for String {
    fn from(s: &'a str) -> Self {
        String::from(s.to_string())
    }
}

impl From<i32> for String {
    fn from(i: i32) -> Self {
        String::from(i.to_string())
    }
}

impl From<std::string::String> for String {
    fn from(s: std::string::String) -> Self {
        String::Heap(Cow::new(s))
    }
}
