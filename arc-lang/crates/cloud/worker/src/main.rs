// #![feature(arc_unwrap_or_clone)]
// #![allow(unused)]

mod coordinator_connector;
mod coordinator_rx;
mod coordinator_tx;
mod runtime_rx;
mod runtime_tx;
mod server;

use clap::Parser;
use io::config::DEFAULT_COORDINATOR_TCP_ADDR;
use io::socket::parse_addr;
use server::Server;
use std::net::SocketAddr;

#[derive(Parser)]
pub struct Args {
    #[clap(short, long, value_parser = parse_addr, default_value = DEFAULT_COORDINATOR_TCP_ADDR)]
    coordinator: SocketAddr,
}

pub const TARGET: &str = env!("TARGET");

#[tokio::main(flavor = "current_thread")]
async fn main() {
    io::tracing::init();
    tracing::info!("Starting worker on {TARGET}");
    Server::start(Args::parse()).await;
}
