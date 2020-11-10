#![allow(clippy::useless_attribute)]

use derive_more::Constructor;
use serde::Deserialize;
use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Deserialize, Debug, Constructor, Clone)]
pub struct Connector {
    name: String,
    endpoint: Endpoint,
    provider: Provider,
}

#[derive(Deserialize, Debug, Clone)]
pub enum Endpoint {
    Source,
    Sink,
}

#[derive(Deserialize, Debug, Clone)]
pub enum Provider {
    Socket(SocketAddr),
    File(PathBuf),
}
