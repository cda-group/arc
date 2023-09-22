use std::net::ToSocketAddrs;

use serde::Deserialize;
use serde::Serialize;

use crate::string::String;

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[repr(C)]
pub struct SocketAddr(pub std::net::SocketAddr);

impl SocketAddr {
    pub fn new(ip: &'static str, port: u16) -> Self {
        Self(std::net::SocketAddr::new(ip.parse().unwrap(), port))
    }
    pub fn parse(addr: String) -> Self {
        addr.as_ref()
            .to_socket_addrs()
            .unwrap()
            .next()
            .map(Self)
            .unwrap()
    }
}
