use arrayvec::ArrayVec;
use futures::SinkExt;
use tokio::net::tcp::OwnedWriteHalf;
use tokio::net::TcpStream;
use tokio_serde::formats::Json;
use tokio_serde::Framed;
use tokio_util::codec::FramedWrite;
use tokio_util::codec::LengthDelimitedCodec;

use crate::data::Data;
use crate::event::Event;
use crate::event::NetEvent;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;

pub struct ShuffleSink<const N: usize, T, K> {
    sinks: ArrayVec<Sender<NetEvent<T>>, N>,
    fun: fn(T) -> K,
}

impl<const N: usize, T: Data, K: Data + Hash> ShuffleSink<N, T, K> {
    pub async fn new(addrs: [&str; N], fun: fn(T) -> K) -> Self {
        let mut sinks = ArrayVec::<Sender<NetEvent<T>>, N>::new();
        for addr in addrs.iter() {
            sinks.push(connect(addr).await);
        }
        Self { sinks, fun }
    }

    pub async fn send(&mut self, iter: impl Iterator<Item = Event<T>>) {
        for event in iter {
            if let Event::Data(time, data) = event {
                let key = (self.fun)(data.clone());
                let mut hasher = DefaultHasher::new();
                key.hash(&mut hasher);
                let hash = hasher.finish() as usize;
                let shard = hash % N;
                self.sinks[shard]
                    .feed(NetEvent::Data(time, data))
                    .await
                    .expect("send");
            }
        }
    }

    pub async fn send_epoch(&mut self, id: u64) {
        for sink in self.sinks.iter_mut() {
            sink.feed(NetEvent::Epoch(id)).await.expect("send");
        }
    }
}

async fn connect<T: Data>(addr: &str) -> Sender<T> {
    let mut stream = TcpStream::connect(addr).await.expect("Failed to connect");
    let (_, tx) = stream.into_split();
    let tx = FramedWrite::new(tx, LengthDelimitedCodec::new());
    let tx = Framed::<_, T, T, _>::new(tx, Json::<T, T>::default());
    tx
}

type Sender<T> = Framed<FramedWrite<OwnedWriteHalf, LengthDelimitedCodec>, T, T, Json<T, T>>;
