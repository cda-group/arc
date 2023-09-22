use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;
use io::config::DEFAULT_COORDINATOR_TCP_ADDR;
use io::socket::parse_addr;
use server::Server;

mod coordinator_connector;
pub mod coordinator_rx;
pub mod coordinator_tx;
pub mod driver_rx;
pub mod driver_tx;
mod server;

#[derive(Parser)]
pub struct Args {
    #[clap(short, long, value_parser = parse_addr, default_value = DEFAULT_COORDINATOR_TCP_ADDR)]
    coordinator: SocketAddr,

    #[clap(short, long)]
    file: PathBuf,
}

#[tokio::main]
async fn main() {
    io::tracing::init();
    Server::new(Args::parse()).await;
}
