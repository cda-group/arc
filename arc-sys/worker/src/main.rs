#![feature(arc_unwrap_or_clone)]
// #![allow(unused)]

mod coordinator_connector;
mod coordinator_receiver;
mod coordinator_sender;
mod dataflow_receiver;
mod dataflow_sender;
mod server;

use clap::Parser;
use server::Server;
use shared::config::DEFAULT_COORDINATOR_TCP_ADDR;
use shared::socket::parse_addr;
use std::net::SocketAddr;

#[derive(Parser)]
pub struct Args {
    #[clap(short, long, value_parser = parse_addr, default_value = DEFAULT_COORDINATOR_TCP_ADDR)]
    coordinator: SocketAddr,
}

pub const TARGET: &str = env!("TARGET");

#[tokio::main(flavor = "current_thread")]
async fn main() {
    shared::tracing::init();
    tracing::info!("Starting worker on {TARGET}");
    Server::start(Args::parse()).await;
}
