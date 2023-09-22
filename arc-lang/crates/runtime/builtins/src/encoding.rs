use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(C)]
pub enum Encoding {
    Csv { sep: char },
    Json,
}

impl Encoding {
    pub fn csv(sep: char) -> Self {
        Self::Csv { sep }
    }
    pub fn json() -> Self {
        Self::Json
    }
}
