use std::net::SocketAddr;

use api::ClientAPI;
use api::CoordinatorAPI;
use futures::SinkExt;
use tokio::net::TcpStream;
use tokio_serde::formats::Json;
use tokio_serde::Framed;
use tokio_util::codec::FramedRead;
use tokio_util::codec::FramedWrite;
use tokio_util::codec::LengthDelimitedCodec;
use io::tcp;

use crate::coordinator_rx::CoordinatorRx;
use crate::coordinator_tx::CoordinatorTx;

use super::server::Server;

struct Actor {
    addr: SocketAddr,
    server: Server,
}

impl Actor {
    fn new(addr: SocketAddr, server: Server) -> Self {
        Self { addr, server }
    }

    async fn run(mut self) {
        tracing::info!("Connecting to {}", self.addr);
        loop {
            match TcpStream::connect(self.addr).await {
                Ok(msg) => {
                    self.handle(msg).await;
                    break;
                }
                Err(err) => {
                    tracing::error!("Connection failed: {}", err);
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                }
            }
        }
    }

    async fn handle(&mut self, stream: TcpStream) {
        let (rx, tx) = stream.into_split();
        let rx = FramedRead::new(rx, LengthDelimitedCodec::new());
        let rx: tcp::Rx<ClientAPI> = Framed::new(rx, Json::default());
        let tx = FramedWrite::new(tx, LengthDelimitedCodec::new());
        let mut tx: tcp::Tx<CoordinatorAPI> = Framed::new(tx, Json::default());
        tx.send(CoordinatorAPI::RegisterClient)
            .await
            .expect("Failed to send register message");
        tracing::info!("Handshake successful");

        self.server.connect(CoordinatorTx::start(tx)).await;
        CoordinatorRx::start(rx, self.server.clone());
    }
}

pub struct CoordinatorConnector;

impl CoordinatorConnector {
    pub fn start(addr: SocketAddr, server: Server) -> Self {
        tokio::spawn(Actor::new(addr, server).run());
        Self
    }
}
