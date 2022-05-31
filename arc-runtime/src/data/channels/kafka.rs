use rskafka::client::partition::Compression;

use rskafka::client::partition::PartitionClient;

use rskafka::client::Client;

use rskafka::record::Record;
use rskafka::record::RecordAndOffset;
use rskafka::time::OffsetDateTime;

use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde::Serializer;
use serde_derive::Deserialize;
use serde_derive::Serialize;

use crate::data::serde::SerdeState;
use crate::dispatch::Execute;

use serde_state::DeserializeState;
use serde_state::SerializeState;

use std::collections::BTreeMap;

use std::collections::VecDeque;
use std::marker::PhantomData;

use crate::control::Control;
use crate::data::gc::Heap;
use crate::data::gc::Trace;
use crate::data::serde::deserialise;
use crate::data::serde::serialise;
use crate::data::Context;
use crate::data::Data;

const BUFFER_SIZE: usize = 100;

#[derive(Debug)]
pub struct PushChan<T: Data> {
    client: PartitionClient,
    topic: String,
    partition: i32,
    buffer: Vec<T>,
    phantom: PhantomData<T>,
}

#[derive(Debug)]
pub struct PullChan<T: Data> {
    client: PartitionClient,
    topic: String,
    partition: i32,
    buffer: VecDeque<RecordAndOffset>,
    offset: i64,
    high_watermark: i64,
    phantom: PhantomData<T>,
}

impl<T: Data> Trace for PullChan<T> {
    fn trace(&self, heap: Heap) {}

    fn root(&self, heap: Heap) {}

    fn unroot(&self, heap: Heap) {}

    fn copy(&self, heap: Heap) -> Self {
        unreachable!("Tried to copy a PullChan to a new heap")
    }
}

impl<T: Data> Trace for PushChan<T> {
    fn trace(&self, heap: Heap) {}

    fn root(&self, heap: Heap) {}

    fn unroot(&self, heap: Heap) {}

    fn copy(&self, heap: Heap) -> Self {
        unreachable!("Tried to copy a PushChan to a new heap")
    }
}

impl<T: Data> PushChan<T> {
    fn new(topic: &str, partition: i32, client: &Client) -> Self {
        Self {
            client: futures::executor::block_on(client.partition_client(topic, partition)).unwrap(),
            topic: topic.to_string(),
            partition,
            buffer: Vec::with_capacity(BUFFER_SIZE),
            phantom: PhantomData,
        }
    }
}

impl<T: Data> PullChan<T> {
    fn new(topic: &str, partition: i32, client: &Client) -> Self {
        Self {
            client: futures::executor::block_on(client.partition_client(topic, partition)).unwrap(),
            topic: topic.to_string(),
            partition,
            buffer: VecDeque::with_capacity(BUFFER_SIZE),
            offset: 0,
            high_watermark: 0,
            phantom: PhantomData,
        }
    }
    fn load(
        topic: String,
        partition: i32,
        buffer: VecDeque<RecordAndOffset>,
        offset: i64,
        high_watermark: i64,
        client: &Client,
    ) -> Self {
        Self {
            client: futures::executor::block_on(client.partition_client(&topic, partition))
                .unwrap(),
            topic,
            partition,
            buffer,
            offset,
            high_watermark,
            phantom: PhantomData,
        }
    }
}

pub fn channel<T: Data>(topic: &str, ctx: Context<impl Execute>) -> (PushChan<T>, PullChan<T>) {
    (
        PushChan::new(topic, ctx.partition, ctx.messaging.as_ref()),
        PullChan::new(topic, ctx.partition, ctx.messaging.as_ref()),
    )
}

impl<T: Data> PushChan<T> {
    pub async fn push(&mut self, data: T, ctx: Context<impl Execute>) {
        if self.buffer.len() == self.buffer.capacity() {
            let records = self
                .buffer
                .drain(..)
                .map(|x| value_to_record(x, ctx.serde))
                .collect::<Vec<_>>();
            self.client
                .produce(records, Compression::default())
                .await
                .unwrap();
        } else {
            self.buffer.push(data);
        }
    }
}

impl<T: Data> PullChan<T> {
    pub async fn pull(&mut self, ctx: Context<impl Execute>) -> T {
        if self.buffer.is_empty() {
            loop {
                let (mut records, high_watermark) = self
                    .client
                    .fetch_records(self.offset, 1..1_000_000, 1_000)
                    .await
                    .unwrap();
                if !records.is_empty() {
                    self.buffer.extend(records.drain(..));
                    self.high_watermark = high_watermark;
                    break;
                }
            }
        }
        let pair = self.buffer.pop_back().unwrap();
        self.offset = pair.offset;
        let data = pair.record.value.unwrap();
        deserialise(data, ctx.serde)
    }
}

fn value_to_record<T: Data>(value: T, serde: SerdeState) -> Record {
    Record {
        key: None,
        value: Some(serialise(value, serde)),
        headers: BTreeMap::new(),
        timestamp: OffsetDateTime::now_utc(),
    }
}

fn bytes_to_record(bytes: Vec<u8>) -> Record {
    Record {
        key: None,
        value: Some(bytes),
        headers: BTreeMap::new(),
        timestamp: OffsetDateTime::now_utc(),
    }
}

#[derive(Serialize, Deserialize)]
struct PushChanSerde {
    topic: String,
    partition: i32,
}

#[derive(Serialize, Deserialize)]
struct PullChanSerde {
    topic: String,
    partition: i32,
    offset: i64,
    high_watermark: i64,
    buffer: Vec<SerdeRecord>,
}

#[derive(Serialize, Deserialize)]
struct SerdeRecord {
    value: Vec<u8>,
    offset: i64,
}

impl<T: Data> SerializeState<SerdeState> for PushChan<T> {
    fn serialize_state<S: Serializer>(&self, s: S, _state: &SerdeState) -> Result<S::Ok, S::Error> {
        PushChanSerde {
            topic: self.topic.clone(),
            partition: self.partition,
        }
        .serialize(s)
    }
}

impl<'de, T: Data> DeserializeState<'de, SerdeState> for PushChan<T> {
    fn deserialize_state<D: Deserializer<'de>>(
        ctx: &mut SerdeState,
        d: D,
    ) -> Result<Self, D::Error> {
        let PushChanSerde { topic, partition } = PushChanSerde::deserialize(d)?;
        Ok(PushChan::new(&topic, partition, &ctx.messaging))
    }
}

impl<T: Data> SerializeState<SerdeState> for PullChan<T> {
    fn serialize_state<S: Serializer>(&self, s: S, _state: &SerdeState) -> Result<S::Ok, S::Error> {
        PullChanSerde {
            topic: self.topic.clone(),
            partition: self.partition,
            offset: self.offset,
            high_watermark: self.high_watermark,
            buffer: self
                .buffer
                .iter()
                .map(|pair| SerdeRecord {
                    value: pair.record.value.clone().unwrap(),
                    offset: pair.offset,
                })
                .collect(),
        }
        .serialize(s)
    }
}

impl<'de, T: Data> DeserializeState<'de, SerdeState> for PullChan<T> {
    fn deserialize_state<D: Deserializer<'de>>(
        ctx: &mut SerdeState,
        d: D,
    ) -> Result<Self, D::Error> {
        let PullChanSerde {
            topic,
            partition,
            offset,
            high_watermark,
            buffer,
        } = PullChanSerde::deserialize(d)?;
        let buffer = buffer
            .into_iter()
            .map(|r| RecordAndOffset {
                offset: r.offset,
                record: bytes_to_record(r.value),
            })
            .collect();
        Ok(PullChan::load(
            topic,
            partition,
            buffer,
            offset,
            high_watermark,
            &ctx.messaging,
        ))
    }
}
