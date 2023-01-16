use serde::Deserialize;
use serde::Serialize;
use time::OffsetDateTime;

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum Event<T> {
    Data(Option<OffsetDateTime>, T),
    Watermark(OffsetDateTime),
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum NetEvent<T> {
    Data(Option<OffsetDateTime>, T),
    Epoch(u64),
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum WorkerEvent {
    Commit(u64),
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum PipelineEvent {
    Epoch(u64),
}

impl<T> Event<T> {
    pub fn map<U, F>(self, f: F) -> Event<U>
    where
        F: FnOnce(T) -> U,
    {
        match self {
            Event::Data(time, data) => Event::Data(time, f(data)),
            Event::Watermark(time) => Event::Watermark(time),
        }
    }
}
