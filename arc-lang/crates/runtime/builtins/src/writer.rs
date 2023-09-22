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
pub enum Writer {
    Stdout,
    File { path: Path },
    Http { url: Url },
    Tcp { addr: SocketAddr },
    Kafka { addr: SocketAddr, topic: String },
}

impl Writer {
    pub fn stdout() -> Self {
        Self::Stdout
    }
    pub fn file(path: Path) -> Self {
        Self::File { path }
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
