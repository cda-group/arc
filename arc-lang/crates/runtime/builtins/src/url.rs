use serde::Deserialize;
use serde::Serialize;

use crate::result::Result;
use crate::string::String;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(C)]
pub struct Url(pub url::Url);

impl Url {
    pub fn parse(s: String) -> Result<Self> {
        match url::Url::parse(s.as_ref()) {
            Ok(v) => Result::ok(Url(v)),
            Err(s) => Result::error(s.to_string().into()),
        }
    }

    pub fn to_string(self) -> String {
        String::from(self.0.to_string())
    }
}
