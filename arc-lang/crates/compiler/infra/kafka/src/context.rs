#![allow(unused)]

use std::time::Duration;

use anyhow::Result;
use rdkafka::config::ClientConfig;
use rdkafka::consumer::stream_consumer::StreamConsumer;
use rdkafka::consumer::Consumer;
use rdkafka::producer::FutureProducer;
use rdkafka::util::Timeout;

pub struct Context {
    pub consumer: StreamConsumer,
    pub producer: FutureProducer,
    pub broker: String,
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context").finish()
    }
}

impl Context {
    pub fn new(broker: Option<String>) -> Result<Self> {
        let broker = broker.unwrap_or("localhost:9092".to_string());
        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", broker.clone())
            .create()?;
        let consumer: StreamConsumer = ClientConfig::new()
            .set("group.id", "kafka-cli")
            .set("bootstrap.servers", broker.clone())
            .create()?;
        let metadata =
            consumer.fetch_metadata(None, Timeout::After(Duration::from_millis(1000)))?;
        Ok(Self {
            consumer,
            producer,
            broker,
        })
    }

    pub fn list(&self) -> Result<()> {
        let metadata = self
            .consumer
            .fetch_metadata(None, Timeout::After(Duration::from_millis(1000)))?;

        eprintln!("Kafka Topics:");
        for topic in metadata.topics() {
            eprintln!(
                "* {} {} partition(s)",
                topic.name(),
                topic.partitions().len()
            );
        }
        Ok(())
    }
}
