//! A module for representing metadata of sources and sinks.

#![allow(clippy::useless_attribute)]

use derive_more::Constructor;
use serde::Deserialize;
use std::net::SocketAddr;
use std::path::PathBuf;

/// A connector is a generalisation of sources and sinks.
#[derive(Deserialize, Debug, Constructor, Clone)]
pub struct Connector {
    /// Name of the connector.
    name: String,
    /// Endpoint of the connector.
    endpoint: Endpoint,
    /// Provider of the connector.
    provider: Provider,
}

/// An kind of endpoint.
#[derive(Deserialize, Debug, Clone)]
pub enum Endpoint {
    /// An endpoint which sends data into the system.
    Source,
    /// An endpoint which reads data from the system.
    Sink,
}

/// An kind of provider.
#[derive(Deserialize, Debug, Clone)]
pub enum Provider {
    /// A provider which reads or writes data to or from a socket.
    Socket(SocketAddr),
    /// A provider which reads or writes data to or from a file.
    File(PathBuf),
}
