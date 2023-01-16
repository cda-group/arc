use std::collections::hash_map::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::marker::PhantomData;

use arrayvec::ArrayVec;
use futures::Sink;
use futures::SinkExt;

use futures::future::join_all;
use futures::stream::SelectAll;
use futures::stream::Stream;
use tokio::net::tcp::OwnedReadHalf;
use tokio::net::tcp::OwnedWriteHalf;
use tokio::net::TcpListener;
use tokio::net::TcpStream;
use tokio_serde::formats::Json;
use tokio_serde::Framed;
use tokio_stream::StreamExt;
use tokio_util::codec::FramedRead;
use tokio_util::codec::FramedWrite;
use tokio_util::codec::LengthDelimitedCodec;

use crate::data::Data;
use crate::event::Event;
use crate::event::NetEvent;

pub struct ShuffleSource<T: Data, K: Data> {
    stream: SelectAll<Receiver<NetEvent<T>>>,
    fun: fn(T) -> K,
}

impl<T: Data, K: Data> ShuffleSource<T, K> {
    pub async fn new(port: u16, num_connections: usize, fun: fn(T) -> K) -> Self {
        let socket = tokio::net::TcpListener::bind(("0.0.0.0", port))
            .await
            .unwrap();
        let connections = (0..num_connections)
            .into_iter()
            .map(|_| connect::<NetEvent<T>>(&socket));
        let connections = futures::future::join_all(connections).await;
        let stream = futures::stream::select_all(connections);

        Self { stream, fun }
    }

    pub async fn recv(&mut self) -> Option<(K, impl Iterator<Item = Event<T>>)> {
        let NetEvent::Data(time, event) = self.stream.next().await?.ok()? else { panic!("TODO") };
        let key = (self.fun)(event.clone());
        Some((key, std::iter::once(Event::Data(time, event))))
    }
}

async fn connect<T: Data>(socket: &TcpListener) -> Receiver<T> {
    let (mut stream, _) = socket.accept().await.expect("Failed to accept");
    let (rx, _) = stream.into_split();
    let rx = FramedRead::new(rx, LengthDelimitedCodec::new());
    let rx = Framed::<_, T, T, _>::new(rx, Json::<T, T>::default());
    rx
}

type Receiver<T> = Framed<FramedRead<OwnedReadHalf, LengthDelimitedCodec>, T, T, Json<T, T>>;
