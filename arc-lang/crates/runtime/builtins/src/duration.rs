use serde::Deserialize;
use serde::Serialize;

#[derive(Clone, Copy, Serialize, Deserialize, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(C)]
pub struct Duration(pub time::Duration);

impl Duration {
    pub fn from_seconds(seconds: i64) -> Self {
        Self(time::Duration::seconds(seconds))
    }

    pub fn from_milliseconds(milliseconds: i64) -> Self {
        Self(time::Duration::milliseconds(milliseconds))
    }

    pub fn from_microseconds(microseconds: i64) -> Self {
        Self(time::Duration::microseconds(microseconds))
    }

    pub fn from_nanoseconds(nanoseconds: i64) -> Self {
        Self(time::Duration::nanoseconds(nanoseconds))
    }

    pub(crate) fn to_std(self) -> std::time::Duration {
        let whole_seconds = self.0.whole_seconds() as u64;
        let subsec_nanos = self.0.subsec_nanoseconds() as u32;
        std::time::Duration::new(whole_seconds, subsec_nanos)
    }
}
