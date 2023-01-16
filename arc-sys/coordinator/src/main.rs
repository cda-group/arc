#![feature(arbitrary_self_types)]
#![feature(hash_drain_filter)]
// #![allow(unused)]

pub mod client_receiver;
pub mod client_sender;
pub mod compiler;
pub mod rest_listener;
pub mod server;
pub mod tls;
pub mod worker_listener;
pub mod worker_receiver;
pub mod worker_sender;

use clap::Parser;
use server::Server;
use shared::config::DEFAULT_COORDAINTOR_REST_PORT;
use shared::config::DEFAULT_COORDINATOR_BROKER_ADDR;
use shared::config::DEFAULT_COORDINATOR_TCP_PORT;
use shared::socket::parse_addr;
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
    which::which("arc-mlir").expect("arc-mlir not found in PATH");
    shared::tracing::init();
    Server::start(Args::parse()).await;
}
