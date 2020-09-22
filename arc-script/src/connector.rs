use derive_more::Constructor;
use serde::Deserialize;
use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Deserialize, Debug, Constructor)]
pub struct Connector {
    name: String,
    endpoint: Endpoint,
    provider: Provider,
}

#[derive(Deserialize, Debug)]
pub enum Endpoint {
    Source,
    Sink,
}

#[derive(Deserialize, Debug)]
pub enum Provider {
    Socket(SocketAddr),
    File(PathBuf),
}

