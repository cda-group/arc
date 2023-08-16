use serde::Deserialize;
use serde::Serialize;

use crate::duration::Duration;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(C)]
pub enum Discretizer {
    Tumbling { length: Duration },
    Sliding { length: Duration, step: Duration },
    Session { gap: Duration },
    Counting { length: i32 },
    Moving { length: i32, step: i32 },
}

impl Discretizer {
    pub fn tumbling(length: Duration) -> Self {
        Self::Tumbling { length }
    }

    pub fn sliding(length: Duration, step: Duration) -> Self {
        Self::Sliding { length, step }
    }

    pub fn session(gap: Duration) -> Self {
        Self::Session { gap }
    }

    pub fn counting(length: i32) -> Self {
        Self::Counting { length }
    }

    pub fn moving(length: i32, step: i32) -> Self {
        Self::Moving { length, step }
    }
}
