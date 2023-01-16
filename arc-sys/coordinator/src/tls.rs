#![allow(unused)]

use rustls::Certificate;
use rustls::PrivateKey;
use shared::api::CoordinatorAPI;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;
use tokio::io;
use tokio::io::split;
use tokio::io::ReadHalf;
use tokio::io::WriteHalf;
use tokio::net::TcpStream;
use tokio_rustls::server::TlsStream;
use tokio_rustls::TlsAcceptor;
use tokio_serde::formats::Json;
use tokio_serde::Framed;
use tokio_util::codec::FramedRead;
use tokio_util::codec::FramedWrite;
use tokio_util::codec::LengthDelimitedCodec;

use shared::api::WorkerAPI;

pub type TlsReceiver<T> =
    Framed<FramedRead<ReadHalf<TlsStream<TcpStream>>, LengthDelimitedCodec>, T, T, Json<T, T>>;

pub type TlsSender<T> =
    Framed<FramedWrite<WriteHalf<TlsStream<TcpStream>>, LengthDelimitedCodec>, T, T, Json<T, T>>;

pub async fn tls_stream(
    acceptor: TlsAcceptor,
    stream: TcpStream,
) -> (TlsReceiver<CoordinatorAPI>, TlsSender<WorkerAPI>) {
    let mut stream = acceptor
        .accept(stream)
        .await
        .expect("Failed to accept TLS stream");
    let (rx, tx) = split(stream);
    let rx = FramedRead::new(rx, LengthDelimitedCodec::new());
    let rx = Framed::new(rx, Json::default());
    let tx = FramedWrite::new(tx, LengthDelimitedCodec::new());
    let tx = Framed::new(tx, Json::default());
    (rx, tx)
}

fn tls_acceptor() -> Result<TlsAcceptor, io::Error> {
    let certs = load_certs(Path::new("todo"))?;
    let mut keys = load_keys(Path::new("todo"))?;

    rustls::ServerConfig::builder()
        .with_safe_defaults()
        .with_no_client_auth()
        .with_single_cert(certs, keys.remove(0))
        .map(|config| TlsAcceptor::from(Arc::new(config)))
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))
}

fn load_certs(path: &Path) -> io::Result<Vec<Certificate>> {
    rustls_pemfile::certs(&mut BufReader::new(File::open(path)?))
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid cert"))
        .map(|mut certs| certs.drain(..).map(Certificate).collect())
}

fn load_keys(path: &Path) -> io::Result<Vec<PrivateKey>> {
    rustls_pemfile::rsa_private_keys(&mut BufReader::new(File::open(path)?))
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid key"))
        .map(|mut keys| keys.drain(..).map(PrivateKey).collect())
}
