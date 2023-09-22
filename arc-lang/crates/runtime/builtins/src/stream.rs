use formats::Decode;
use formats::Encode;
use futures::SinkExt;
use futures::StreamExt;
use time::OffsetDateTime;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;
use tokio::io::BufWriter;
use tokio::sync::mpsc::Receiver;

use crate::duration::Duration;
use crate::encoding::Encoding;
use crate::keyed_stream::KeyedEvent;
use crate::keyed_stream::KeyedStream;
use crate::path::Path;
use crate::reader::Reader;
use crate::socket::SocketAddr;
use crate::time::Time;
use crate::time_source::TimeSource;
use crate::traits::Data;
use crate::url::Url;
use crate::writer::Writer;
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Serialize, Deserialize)]
#[repr(C)]
pub(crate) enum Event<T> {
    Data(Time, T),
    Watermark(Time),
    Snapshot(usize),
    Sentinel,
}

pub struct Stream<T: Data>(pub(crate) Receiver<Event<T>>);

impl<T: Data> Stream<T> {
    pub fn source(
        reader: Reader,
        encoding: Encoding,
        time_source: TimeSource<fn(T) -> Time>,
    ) -> Stream<T> {
        Self::_source_encoding(reader, encoding, time_source)
    }

    fn _source_encoding(
        reader: Reader,
        encoding: Encoding,
        time_source: TimeSource<fn(T) -> Time>,
    ) -> Stream<T> {
        match encoding {
            Encoding::Csv { sep } => {
                let decoder = formats::csv::de::Reader::<1024>::new(sep);
                Self::_source_reader(reader, decoder, time_source)
            }
            Encoding::Json => {
                let decoder = formats::json::de::Reader::new();
                Self::_source_reader(reader, decoder, time_source)
            }
        }
    }

    async fn read_pipe(
        rx: impl AsyncReadExt + Unpin,
        mut decoder: impl Decode + 'static,
        watch: bool,
        tx: tokio::sync::mpsc::Sender<T>,
    ) {
        let mut rx = BufReader::new(rx);
        let mut buf = Vec::with_capacity(1024);
        loop {
            match rx.read_until(b'\n', &mut buf).await {
                Ok(0) => {
                    tracing::info!("EOF");
                    if watch {
                        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    } else {
                        break;
                    }
                }
                Ok(n) => match decoder.decode(&buf[0..n]) {
                    Ok(data) => {
                        tracing::info!("Decoded: {:?}", data);
                        tx.send(data).await.unwrap();
                        buf.clear();
                    }
                    Err(e) => tracing::info!("Failed to decode: {}", e),
                },
                Err(e) => panic!("Failed to read from stdin: {}", e),
            }
        }
    }

    async fn read_file(
        path: Path,
        decoder: impl Decode + 'static,
        watch: bool,
        tx2: tokio::sync::mpsc::Sender<T>,
    ) {
        match tokio::fs::File::open(&path.0).await {
            Ok(rx) => Self::read_pipe(rx, decoder, watch, tx2).await,
            Err(e) => panic!("Failed to open file `{}`: {}", path.0.display(), e),
        }
    }

    async fn read_socket(
        addr: SocketAddr,
        mut decoder: impl Decode + 'static,
        tx: tokio::sync::mpsc::Sender<T>,
    ) {
        tracing::info!("Trying to listen on {}", addr.0);
        let socket = tokio::net::TcpListener::bind(addr.0).await.unwrap();
        tracing::info!("Listening on {}", addr.0);
        let (socket, _) = socket.accept().await.unwrap();
        tracing::info!("Accepted connection from {}", addr.0);
        let mut rx = tokio_util::codec::Framed::new(socket, tokio_util::codec::LinesCodec::new());
        loop {
            match rx.next().await {
                Some(Ok(line)) => match decoder.decode(line.as_bytes()) {
                    Ok(data) => {
                        tracing::info!("Decoded: {:?}", data);
                        tx.send(data).await.unwrap()
                    }
                    Err(e) => tracing::info!("Failed to decode: {}", e),
                },
                Some(Err(e)) => tracing::info!("Failed to read: {}", e),
                None => break,
            }
        }
    }

    #[allow(unused)]
    async fn read_http(url: Url, decoder: impl Decode + 'static, tx: tokio::sync::mpsc::Sender<T>) {
        todo!()
        // let uri: Uri = url.0.to_string().parse().unwrap();
        // let client = hyper::Client::new();
        // let mut resp = client.get(uri).await.unwrap();
        // loop {
        //     match resp.body_mut().data().await {
        //         Some(Ok(chunk)) => match decoder.decode(&chunk) {
        //             Ok(data) => {
        //                 tracing::info!("Decoded: {:?}", data);
        //                 tx.send(data).await.unwrap();
        //             }
        //             Err(e) => tracing::info!("Failed to decode: {}", e),
        //         },
        //         Some(Err(e)) => tracing::info!("Failed to read: {}", e),
        //         None => break,
        //     }
        // }
    }

    #[allow(unused)]
    async fn write_http(
        rx: tokio::sync::mpsc::Receiver<T>,
        url: Url,
        encoder: impl Encode + 'static,
    ) {
        todo!()
        // let uri: Uri = url.0.to_string().parse().unwrap();
        // let client = hyper::Client::new();
        // let (mut tx1, rx1) = futures::channel::mpsc::channel(100);
        // let req = Request::builder()
        //     .method(Method::POST)
        //     .uri(uri)
        //     .header("content-type", encoder.content_type())
        //     .body(Body::wrap_stream(rx1))
        //     .unwrap();
        // client.request(req).await.unwrap();
        // let mut buf = vec![0; 1024];
        // loop {
        //     match rx.recv().await {
        //         Some(data) => match encoder.encode(&data, &mut buf) {
        //             Ok(n) => {
        //                 tracing::info!("Encoded: {:?}", data);
        //                 let bytes: Result<_, std::io::Error> =
        //                     Ok(hyper::body::Bytes::from(buf[0..n].to_vec()));
        //                 tx1.send(bytes).await.unwrap();
        //             }
        //             Err(e) => tracing::info!("Failed to encode: {}", e),
        //         },
        //         None => break,
        //     }
        // }
    }

    async fn write_pipe(
        mut rx: tokio::sync::mpsc::Receiver<T>,
        mut encoder: impl Encode + 'static,
        tx: impl AsyncWriteExt + Unpin,
    ) {
        let mut tx = BufWriter::new(tx);
        let mut buf = vec![0; 1024];
        loop {
            match rx.recv().await {
                Some(data) => match encoder.encode(&data, &mut buf) {
                    Ok(n) => {
                        tracing::info!("Encoded: {:?}", data);
                        tx.write_all(&buf[0..n]).await.unwrap();
                        tx.flush().await.unwrap();
                    }
                    Err(e) => tracing::info!("Failed to encode: {}", e),
                },
                None => break,
            }
        }
    }

    async fn write_file(
        rx: tokio::sync::mpsc::Receiver<T>,
        path: Path,
        encoder: impl Encode + 'static,
    ) {
        match tokio::fs::File::create(&path.0).await {
            Ok(tx) => Self::write_pipe(rx, encoder, tx).await,
            Err(e) => panic!("Failed to open file `{}`: {}", path.0.display(), e),
        }
    }

    async fn write_socket(
        mut rx: tokio::sync::mpsc::Receiver<T>,
        addr: SocketAddr,
        mut encoder: impl Encode + 'static,
    ) {
        tracing::info!("Connecting to {}", addr.0);
        let socket = tokio::net::TcpStream::connect(addr.0).await.unwrap();
        tracing::info!("Connected to {}", addr.0);
        let mut tx = tokio_util::codec::Framed::new(socket, tokio_util::codec::LinesCodec::new());
        let mut buf = vec![0; 1024];
        loop {
            match rx.recv().await {
                Some(data) => match encoder.encode(&data, &mut buf) {
                    Ok(n) => {
                        tracing::info!("Encoded: {:?}", data);
                        let s = std::str::from_utf8(&buf[0..n - 1]).unwrap(); // -1 to remove trailing newline
                        tracing::info!("Sending: [{}]", s);
                        tx.send(s).await.unwrap();
                    }
                    Err(e) => tracing::info!("Failed to encode: {}", e),
                },
                None => break,
            }
        }
    }

    fn _source_reader(
        reader: Reader,
        decoder: impl Decode + 'static,
        time_source: TimeSource<fn(T) -> Time>,
    ) -> Stream<T> {
        let (tx2, rx2) = tokio::sync::mpsc::channel(100);
        tokio::task::spawn_local(async move {
            match reader {
                Reader::Stdin => Self::read_pipe(tokio::io::stdin(), decoder, false, tx2).await,
                Reader::File { path, watch } => Self::read_file(path, decoder, watch, tx2).await,
                Reader::Http { url } => Self::read_http(url, decoder, tx2).await,
                Reader::Tcp { addr } => Self::read_socket(addr, decoder, tx2).await,
                Reader::Kafka { addr: _, topic: _ } => todo!(),
            }
        });
        Self::_source_time(rx2, time_source)
    }

    fn _source_time(
        rx: tokio::sync::mpsc::Receiver<T>,
        time_source: TimeSource<fn(T) -> Time>,
    ) -> Stream<T> {
        match time_source {
            TimeSource::Ingestion { watermark_interval } => {
                false;
                Self::_source_ingestion_time(rx, watermark_interval)
            }
            TimeSource::Event {
                extractor,
                watermark_interval,
                slack,
            } => Self::_source_event_time(rx, extractor, watermark_interval, slack),
        }
    }

    fn _source_ingestion_time(
        mut rx: tokio::sync::mpsc::Receiver<T>,
        watermark_interval: Duration,
    ) -> Stream<T> {
        let (tx1, rx1) = tokio::sync::mpsc::channel(100);
        tokio::task::spawn(async move {
            let mut watermark_interval = tokio::time::interval(watermark_interval.to_std());
            loop {
                tokio::select! {
                    _ = watermark_interval.tick() => {
                        tx1.send(Event::Watermark(Time::now())).await.expect("Failed to send watermark");
                    },
                    data = rx.recv() => {
                        match data {
                            Some(data) => tx1.send(Event::Data(Time::now(), data)).await.expect("Failed to send data"),
                            None => {
                                tx1.send(Event::Sentinel).await.expect("Failed to send sentinel");
                                break;
                            },
                        }
                    }
                }
            }
        });
        Stream(rx1)
    }

    fn _source_event_time(
        mut rx: tokio::sync::mpsc::Receiver<T>,
        extractor: fn(T) -> Time,
        watermark_interval: Duration,
        slack: Duration,
    ) -> Stream<T> {
        let (tx1, rx1) = tokio::sync::mpsc::channel(100);
        tokio::task::spawn(async move {
            let mut latest_time = OffsetDateTime::UNIX_EPOCH;
            let slack = slack.to_std();
            let mut watermark_interval = tokio::time::interval(watermark_interval.to_std());
            let mut watermark = OffsetDateTime::UNIX_EPOCH;
            loop {
                tokio::select! {
                    _ = watermark_interval.tick() => {
                        if latest_time > OffsetDateTime::UNIX_EPOCH {
                            watermark = latest_time - slack;
                            tx1.send(Event::Watermark(Time(watermark))).await.expect("Failed to send watermark");
                        }
                    },
                    data = rx.recv() => {
                        match data {
                            Some(data) => {
                                let time = extractor(data.clone());
                                if time.0 < watermark {
                                    continue;
                                }
                                if time.0 > latest_time {
                                    latest_time = time.0;
                                }
                                tx1.send(Event::Data(time, data)).await.expect("Failed to send data");
                            }
                            None => {
                                tx1.send(Event::Sentinel).await.expect("Failed to send sentinel");
                                break;
                            },
                        }
                    }
                }
            }
        });
        Stream(rx1)
    }

    pub fn sink(self, writer: Writer, encoding: Encoding) {
        let mut this = self;
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        tokio::task::spawn_local(async move {
            loop {
                let event = this.0.recv().await.expect("failed to receive event");
                match event {
                    Event::Data(_, data) => tx.send(data).await.unwrap(),
                    Event::Watermark(_) => continue,
                    Event::Snapshot(_) => todo!(),
                    Event::Sentinel => break,
                }
            }
        });
        Self::_sink_encoding(rx, writer, encoding);
    }

    fn _sink_encoding(rx: tokio::sync::mpsc::Receiver<T>, writer: Writer, encoding: Encoding) {
        match encoding {
            Encoding::Csv { sep } => {
                let encoder = formats::csv::ser::Writer::new(sep);
                Self::_sink_writer(rx, writer, encoder);
            }
            Encoding::Json => {
                let encoder = formats::json::ser::Writer::new();
                Self::_sink_writer(rx, writer, encoder);
            }
        }
    }

    fn _sink_writer(
        rx: tokio::sync::mpsc::Receiver<T>,
        writer: Writer,
        encoder: impl Encode + 'static,
    ) {
        tokio::task::spawn_local(async move {
            match writer {
                Writer::Stdout => Self::write_pipe(rx, encoder, tokio::io::stdout()).await,
                Writer::File { path } => Self::write_file(rx, path, encoder).await,
                Writer::Http { url } => Self::write_http(rx, url, encoder).await,
                Writer::Tcp { addr } => Self::write_socket(rx, addr, encoder).await,
                Writer::Kafka { addr: _, topic: _ } => todo!(),
            }
        });
    }

    pub fn map<O>(mut self, f: fn(T) -> O) -> Stream<O>
    where
        O: Data,
    {
        let (tx1, rx1) = tokio::sync::mpsc::channel(100);
        tokio::task::spawn_local(async move {
            loop {
                match self.recv().await {
                    Event::Data(t, v) => tx1.send(Event::Data(t, f(v))).await,
                    Event::Watermark(t) => tx1.send(Event::Watermark(t)).await,
                    Event::Snapshot(i) => tx1.send(Event::Snapshot(i)).await,
                    Event::Sentinel => {
                        tx1.send(Event::Sentinel).await.unwrap();
                        break;
                    }
                }
                .unwrap()
            }
        });
        Stream(rx1)
    }

    pub fn filter(mut self, f: fn(T) -> bool) -> Stream<T> {
        let (tx1, rx1) = tokio::sync::mpsc::channel(100);
        tokio::task::spawn_local(async move {
            loop {
                match self.recv().await {
                    Event::Data(t, v) => {
                        if f(v.clone()) {
                            tx1.send(Event::Data(t, v)).await.unwrap();
                        }
                    }
                    Event::Watermark(t) => tx1.send(Event::Watermark(t)).await.unwrap(),
                    Event::Snapshot(i) => tx1.send(Event::Snapshot(i)).await.unwrap(),
                    Event::Sentinel => {
                        tx1.send(Event::Sentinel).await.unwrap();
                        break;
                    }
                }
            }
        });
        Stream(rx1)
    }

    pub fn keyby<K: Data>(mut self, fun: fn(T) -> K) -> KeyedStream<K, T> {
        let (tx1, rx1) = tokio::sync::mpsc::channel(100);
        tokio::task::spawn_local(async move {
            loop {
                match self.0.recv().await.unwrap() {
                    Event::Data(t, v) => {
                        let k = fun(v.clone());
                        tx1.send(KeyedEvent::Data(t, k, v)).await.unwrap();
                    }
                    Event::Watermark(t) => {
                        tx1.send(KeyedEvent::Watermark(t)).await.unwrap();
                    }
                    Event::Snapshot(i) => {
                        tx1.send(KeyedEvent::Snapshot(i)).await.unwrap();
                    }
                    Event::Sentinel => {
                        tx1.send(KeyedEvent::Sentinel).await.unwrap();
                        break;
                    }
                }
            }
        });
        KeyedStream(rx1)
    }

    pub fn scan<A: Data>(mut self, init: A, fun: fn(T, A) -> A) -> Stream<A> {
        let (tx1, rx1) = tokio::sync::mpsc::channel(100);
        tokio::task::spawn_local(async move {
            let mut acc = init;
            loop {
                match self.0.recv().await.unwrap() {
                    Event::Data(t, v) => {
                        acc = fun(v.clone(), acc);
                        tx1.send(Event::Data(t, acc.clone())).await.unwrap();
                    }
                    Event::Watermark(t) => {
                        tx1.send(Event::Watermark(t)).await.unwrap();
                    }
                    Event::Snapshot(i) => {
                        tx1.send(Event::Snapshot(i)).await.unwrap();
                    }
                    Event::Sentinel => {
                        tx1.send(Event::Sentinel).await.unwrap();
                        break;
                    }
                }
            }
        });
        Stream(rx1)
    }

    pub fn merge(mut self, mut other: Self) -> Self {
        let (tx2, rx2) = tokio::sync::mpsc::channel(100);
        tokio::task::spawn_local(async move {
            loop {
                let event = tokio::select! {
                    event = self.recv() => if let Event::Sentinel = event {
                        other.recv().await
                    } else {
                        event
                    },
                    event = other.recv() => if let Event::Sentinel = event {
                        self.recv().await
                    } else {
                        event
                    },
                };
                match event {
                    Event::Data(t, v1) => {
                        tx2.send(Event::Data(t, v1)).await.unwrap();
                    }
                    Event::Watermark(t) => {
                        tx2.send(Event::Watermark(t)).await.unwrap();
                    }
                    Event::Snapshot(i) => {
                        tx2.send(Event::Snapshot(i)).await.unwrap();
                    }
                    Event::Sentinel => {
                        tx2.send(Event::Sentinel).await.unwrap();
                        break;
                    }
                }
            }
        });
        Self(rx2)
    }

    pub fn split(mut self) -> (Self, Self) {
        let (tx1, rx1) = tokio::sync::mpsc::channel(100);
        let (tx2, rx2) = tokio::sync::mpsc::channel(100);
        tokio::task::spawn_local(async move {
            loop {
                let (l, r) = match self.recv().await {
                    Event::Data(t, v1) => {
                        let v2 = v1.clone();
                        tokio::join!(tx1.send(Event::Data(t, v2)), tx2.send(Event::Data(t, v1)),)
                    }
                    Event::Watermark(t) => {
                        tokio::join!(tx1.send(Event::Watermark(t)), tx2.send(Event::Watermark(t)))
                    }
                    Event::Snapshot(i) => {
                        tokio::join!(tx1.send(Event::Snapshot(i)), tx2.send(Event::Snapshot(i)))
                    }
                    Event::Sentinel => {
                        tokio::join!(tx1.send(Event::Sentinel), tx2.send(Event::Sentinel))
                    }
                };
                l.unwrap();
                r.unwrap();
            }
        });
        (Self(rx1), Self(rx2))
    }

    async fn recv(&mut self) -> Event<T> {
        self.0.recv().await.unwrap()
    }
}
