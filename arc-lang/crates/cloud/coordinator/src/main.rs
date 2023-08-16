// #![feature(arbitrary_self_types)]

pub mod client_listener;
pub mod client_rx;
pub mod client_tx;
pub mod rest_listener;
pub mod server;
pub mod tls;
pub mod worker_listener;
pub mod worker_rx;
pub mod worker_tx;

use clap::Parser;
use io::config::DEFAULT_COORDAINTOR_REST_PORT;
use io::config::DEFAULT_COORDINATOR_BROKER_ADDR;
use io::config::DEFAULT_COORDINATOR_TCP_PORT;
use io::socket::parse_addr;
use server::Server;
use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Parser, Debug)]
pub struct Args {
    #[clap(short, long, default_value_t = DEFAULT_COORDINATOR_TCP_PORT)]
    tcp_listener_port: u16,

    #[clap(short, long, default_value_t = DEFAULT_COORDAINTOR_REST_PORT)]
    rest_listener_port: u16,

    #[clap(short, long, value_parser = parse_addr, default_value = DEFAULT_COORDINATOR_BROKER_ADDR)]
    broker: SocketAddr,

    #[clap(long)]
    certificate: Option<PathBuf>,

    #[clap(long)]
    key: Option<PathBuf>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // which::which("arc-mlir").expect("arc-mlir not found in PATH");
    io::tracing::init();
    Server::start(Args::parse()).await;
}
