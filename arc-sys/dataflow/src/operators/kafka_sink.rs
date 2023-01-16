use std::time::Duration;

use crate::data::Data;
use crate::event::Event;
use rdkafka::producer::FutureProducer;
use rdkafka::producer::FutureRecord;
use rdkafka::ClientConfig;

pub struct KafkaSink<'a, T, K> {
    producer: FutureProducer,
    topic: &'a str,
    fun: fn(T) -> K,
}

/// `brokers` are a comma-separated list of brokers ("IP:PORT,IP:PORT,...").
pub fn connect(brokers: &str) -> FutureProducer {
    ClientConfig::new()
        .set("bootstrap.servers", brokers)
        .set("message.timeout.ms", "5000")
        .create()
        .expect("Failed to create Kafka producer client")
}

impl<'a, T: Data, K: Data> KafkaSink<'a, T, K> {
    pub fn new(brokers: &str, topic: &'a str, fun: fn(T) -> K) -> Self {
        let producer = connect(brokers);
        Self {
            producer,
            topic,
            fun,
        }
    }

    pub async fn send(&mut self, iter: impl Iterator<Item = Event<T>>) {
        for event in iter {
            if let Event::Data(time, data) = event {
                let key = (self.fun)(data.clone());
                self.producer
                    .send(
                        FutureRecord::to(self.topic)
                            .payload(&serde_json::to_vec(&data).unwrap())
                            .key(&serde_json::to_vec(&key).unwrap()),
                        Duration::from_secs(0),
                    )
                    .await;
            }
        }
    }
}
