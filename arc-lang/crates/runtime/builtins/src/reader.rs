#![allow(unused)]
use serde::Deserialize;
use serde::Serialize;

use crate::path::Path;
use crate::socket::SocketAddr;
use crate::stream::Stream;
use crate::string::String;
use crate::traits::Data;
use crate::url::Url;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub enum Reader {
    Stdin,
    File { path: Path, watch: bool },
    Http { url: Url },
    Tcp { addr: SocketAddr },
    Kafka { addr: SocketAddr, topic: String },
}

impl Reader {
    pub fn stdin() -> Self {
        Self::Stdin
    }
    pub fn file(path: Path, watch: bool) -> Self {
        if !path.0.exists() {
            tracing::warn!("{} does not exist", path.0.display());
        }
        Self::File { path, watch }
    }
    pub fn http(url: Url) -> Self {
        Self::Http { url }
    }
    pub fn tcp(addr: SocketAddr) -> Self {
        Self::Tcp { addr }
    }
    pub fn kafka(addr: SocketAddr, topic: String) -> Self {
        Self::Kafka { addr, topic }
    }
}
