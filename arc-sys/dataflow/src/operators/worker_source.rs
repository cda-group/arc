use futures::StreamExt;
use tokio::io::AsyncRead;
use tokio::io::Stdin;
use tokio_serde::formats::Json;
use tokio_serde::Framed;
use tokio_util::codec::FramedRead;
use tokio_util::codec::LengthDelimitedCodec;

use crate::event::PipelineEvent;
use crate::event::WorkerEvent;

pub struct WorkerSource {
    rx: Receiver<PipelineEvent>,
}

impl WorkerSource {
    pub fn new() -> Self {
        Self { rx: connect() }
    }

    pub async fn recv(&mut self) -> Option<PipelineEvent> {
        self.rx.next().await?.ok()
    }
}

fn connect<T>() -> Receiver<T> {
    let rx = tokio::io::stdin();
    let rx = FramedRead::new(rx, LengthDelimitedCodec::new());
    let rx = Framed::<_, T, T, _>::new(rx, Json::<T, T>::default());
    rx
}

type Receiver<T> = Framed<FramedRead<Stdin, LengthDelimitedCodec>, T, T, Json<T, T>>;
