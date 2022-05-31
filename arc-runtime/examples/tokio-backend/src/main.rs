use futures::sink::SinkExt;
use serde::Deserialize;
use serde::Serialize;
use tokio::net::TcpListener;
use tokio::net::TcpStream;
use tokio::task;
use tokio_serde::formats::SymmetricalJson;
use tokio_serde::SymmetricallyFramed;
use tokio_stream::StreamExt;
use tokio_util::codec::Framed;
use tokio_util::codec::LengthDelimitedCodec;

#[derive(Serialize, Deserialize, Debug)]
struct Message {
    data: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let local = task::LocalSet::new();

    match std::env::args().nth(1).as_deref() {
        Some("ping") => local.run_until(ping()).await?,
        Some("pong") => local.run_until(pong()).await?,
        _ => {
            println!("Usage: {} [ping|pong]", std::env::args().nth(0).unwrap());
            std::process::exit(1);
        }
    };
    Ok(())
}

const PING: &str = "localhost:8082";

async fn ping() -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind(PING).await?;
    let (stream, _) = listener.accept().await?;

    let stream = Framed::new(stream, LengthDelimitedCodec::new());
    let mut stream = SymmetricallyFramed::new(stream, SymmetricalJson::<Message>::default());

    loop {
        stream
            .send(Message {
                data: "ping".to_owned(),
            })
            .await?;
        if let Ok(x) = stream.next().await.unwrap() {
            println!("{x:?}");
        }
    }
}

async fn pong() -> Result<(), Box<dyn std::error::Error>> {
    let stream = TcpStream::connect(PING).await?;
    let stream = Framed::new(stream, LengthDelimitedCodec::new());
    let mut stream = SymmetricallyFramed::new(stream, SymmetricalJson::<Message>::default());

    loop {
        stream
            .send(Message {
                data: "ping".to_owned(),
            })
            .await?;
        if let Ok(x) = stream.next().await.unwrap() {
            println!("{x:?}");
        }
    }
}
