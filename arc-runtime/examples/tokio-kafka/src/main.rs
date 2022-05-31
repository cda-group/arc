use futures::sink::SinkExt;
use rskafka::client::controller::ControllerClient;
use rskafka::client::partition::Compression;
use rskafka::client::Client;
use rskafka::client::ClientBuilder;
use rskafka::record::Record;
use rskafka::time::OffsetDateTime;
use serde::Deserialize;
use serde::Serialize;
use std::collections::BTreeMap;
use tokio::net::TcpListener;
use tokio::net::TcpStream;
use tokio::task;
use tokio_serde::formats::SymmetricalJson;
use tokio_serde::SymmetricallyFramed;
use tokio_stream::StreamExt;
use tokio_util::codec::Framed;
use tokio_util::codec::LengthDelimitedCodec;

struct Context {
    client: Client,
    controller: ControllerClient,
    topic: String,
}

async fn producer(ctx: &Context) -> Result<(), Box<dyn std::error::Error>> {
    let producer = ctx
        .client
        .partition_client(
            ctx.topic.clone(), // topic
            0,                 // partition
        )
        .await?;

    for i in 0..100 {
        let record = Record {
            key: None,
            value: Some(b"hello kafka".to_vec()),
            headers: BTreeMap::new(),
            timestamp: OffsetDateTime::now_utc(),
        };
        producer
            .produce(vec![record], Compression::default())
            .await?;
    }
    Ok(())
}

async fn consumer(ctx: &Context) -> Result<(), Box<dyn std::error::Error>> {
    let consumer = ctx
        .client
        .partition_client(
            ctx.topic.clone(), // topic
            0,                 // partition
        )
        .await?;
    let mut i = 0;
    loop {
        let (records, high_watermark) = consumer
            .fetch_records(
                i,            // offset
                1..1_000_000, // min..max bytes
                1_000,        // max wait time
            )
            .await?;
        if records.is_empty() {
            break;
        } else {
            i += records.len() as i64;
        }
    }
    Ok(())
}

#[derive(Serialize, Deserialize, Debug)]
struct Message {
    data: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let local = task::LocalSet::new();

    // setup client
    let client = ClientBuilder::new(vec!["localhost:9093".to_owned()])
        .build()
        .await?;

    // create a topic
    let topic = "my_topic".to_owned();
    let controller = client.controller_client().await?;

    controller
        .create_topic(
            topic.clone(), // topic
            2,             // partitions
            1,             // replication factor
            5_000,         // timeout (ms)
        )
        .await?;

    let ctx = Context {
        client,
        controller,
        topic,
    };

    match std::env::args().nth(1).as_deref() {
        Some("consumer") => local.run_until(consumer(&ctx)).await?,
        Some("producer") => local.run_until(producer(&ctx)).await?,
        _ => {
            println!("Usage: {} [ping|pong]", std::env::args().nth(0).unwrap());
            std::process::exit(1);
        }
    };
    Ok(())
}
