use serde::Deserialize;
use serde::Serialize;

use crate::duration::Duration;
use crate::time::Time;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub enum TimeSource<F> {
    Ingestion {
        watermark_interval: Duration,
    },
    Event {
        extractor: F,
        watermark_interval: Duration,
        slack: Duration,
    },
}

impl<T> TimeSource<fn(T) -> Time> {
    pub fn ingestion(watermark_interval: Duration) -> Self {
        Self::Ingestion { watermark_interval }
    }
    pub fn event(
        extractor: fn(T) -> Time,
        watermark_interval: Duration,
        slack: Duration,
    ) -> Self {
        Self::Event {
            extractor,
            watermark_interval,
            slack,
        }
    }
}
