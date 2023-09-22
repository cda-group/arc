use std::net::SocketAddr;
use std::net::ToSocketAddrs;

pub fn parse_addr(s: &str) -> Result<SocketAddr, std::io::Error> {
    s.to_socket_addrs()
        .map(|mut addrs| addrs.next().expect("no address"))
}
