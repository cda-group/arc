use futures::SinkExt;
use tokio::io::AsyncWrite;
use tokio::io::Stdout;
use tokio_serde::formats::Json;
use tokio_serde::Framed;
use tokio_util::codec::FramedWrite;
use tokio_util::codec::LengthDelimitedCodec;

use crate::event::WorkerEvent;

pub struct WorkerSink {
    tx: Sender<WorkerEvent>,
}

impl WorkerSink {
    pub fn new() -> Self {
        Self { tx: connect() }
    }

    pub async fn send(&mut self, event: WorkerEvent) {
        self.tx.send(event).await.expect("send");
    }
}

fn connect<T>() -> Sender<T> {
    let tx = tokio::io::stdout();
    let tx = FramedWrite::new(tx, LengthDelimitedCodec::new());
    let tx = Framed::<_, T, T, _>::new(tx, Json::<T, T>::default());
    tx
}

type Sender<T> = Framed<FramedWrite<Stdout, LengthDelimitedCodec>, T, T, Json<T, T>>;
