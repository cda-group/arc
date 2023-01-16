use std::ops::Range;
use std::time::Duration;

use rdkafka::client::Client;
use rdkafka::config::RDKafkaLogLevel;
use rdkafka::consumer::Consumer;
use rdkafka::consumer::ConsumerContext;
use rdkafka::consumer::Rebalance;
use rdkafka::consumer::StreamConsumer;
use rdkafka::error::KafkaError;
use rdkafka::message::BorrowedMessage;
use rdkafka::message::OwnedHeaders;
use rdkafka::producer::FutureProducer;
use rdkafka::producer::FutureRecord;
use rdkafka::producer::ProducerContext;
use rdkafka::ClientConfig;
use rdkafka::ClientContext;
use rdkafka::Message;
use rdkafka::Timestamp;
use rdkafka::TopicPartitionList;
use serde::Deserializer;
use tokio::pin;
use tokio_stream::Stream;
use tokio_stream::StreamExt;

use crate::data::Data;
use crate::event::Event;
use crate::serde::deserialise;

pub struct Context {}

impl ClientContext for Context {}

impl ConsumerContext for Context {}

pub struct KafkaSource<K, T> {
    consumer: StreamConsumer,
    _k: std::marker::PhantomData<K>,
    _t: std::marker::PhantomData<T>,
}

pub fn connect(brokers: &str) -> StreamConsumer {
    ClientConfig::new()
        .set("group.id", "arc-lang")
        .set("bootstrap.servers", brokers)
        .set("enable.partition.eof", "false")
        .set("session.timeout.ms", "6000")
        .set("enable.auto.commit", "true")
        .set_log_level(RDKafkaLogLevel::Debug)
        .create()
        .expect("Failed to create Kafka consumer client")
}

impl<K: Data, T: Data> KafkaSource<K, T> {
    /// `brokers`: A comma-separated list of brokers ("IP:PORT,IP:PORT,...").
    /// `group_id`: The group ID to use for the consumer.
    /// `topics`: A comma-separated list of topics to subscribe to.
    pub fn new(brokers: &str, topic: &str, partitions: Range<i32>) -> Self {
        let consumer = connect(brokers);
        let mut topic_partition_list = TopicPartitionList::new();
        topic_partition_list.add_partition_range(topic, partitions.start, partitions.end);
        consumer.assign(&topic_partition_list);
        Self {
            consumer,
            _k: std::marker::PhantomData,
            _t: std::marker::PhantomData,
        }
    }

    pub async fn recv<'a>(&mut self) -> Option<(Option<K>, impl Iterator<Item = Event<T>>)> {
        let msg = self.consumer.recv().await.ok()?;
        let key = if let Some(key) = msg.key() {
            deserialise(key).ok()
        } else {
            None
        };
        let payload = msg.payload()?;
        let value = deserialise(payload).ok()?;
        let Timestamp::CreateTime(time) = msg.timestamp() else { return None; };
        let time = time::OffsetDateTime::from_unix_timestamp(time).ok()?;
        Some((key, std::iter::once(Event::Data(Some(time), value))))
    }
}
