use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;
use server::Server;
use shared::config::DEFAULT_COORDINATOR_TCP_ADDR;
use shared::socket::parse_addr;

mod coordinator_connector;
pub mod coordinator_receiver;
pub mod coordinator_sender;
// mod rest;
mod server;
pub mod interpreter_receiver;
pub mod interpreter_sender;

#[derive(Parser)]
pub struct Args {
    #[clap(short, long, value_parser = parse_addr, default_value = DEFAULT_COORDINATOR_TCP_ADDR)]
    coordinator: SocketAddr,

    #[clap(short, long)]
    file: PathBuf,
}

#[tokio::main]
async fn main() {
    shared::tracing::init();
    which::which("arc-lang").expect("arc-lang not found in PATH");
    Server::new(Args::parse()).await;
}
