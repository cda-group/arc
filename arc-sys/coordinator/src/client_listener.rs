use shared::api::CoordinatorAPI;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tokio::net::TcpStream;
use tokio_serde::formats::Json;
use tokio_serde::Framed;
use tokio_stream::StreamExt;
use tokio_util::codec::FramedRead;
use tokio_util::codec::FramedWrite;
use tokio_util::codec::LengthDelimitedCodec;

use crate::client_sender::ClientSender;
use crate::server::ClientId;

use super::server::Server;
use super::worker_receiver::WorkerReceiver;

struct Actor {
    port: u16,
    server: Server,
    id: ClientId,
}

impl Actor {
    fn new(port: u16, server: Server) -> Self {
        Self {
            port,
            server,
            id: ClientId(0),
        }
    }

    async fn run(mut self) {
        let addr = SocketAddr::from(([0, 0, 0, 0], self.port));
        // let acceptor = tls_acceptor().expect("Failed to create TLS acceptor");
        let rx = TcpListener::bind(&addr).await.expect("Failed to bind");
        tracing::info!("Listening for clients on {}", addr);
        loop {
            match rx.accept().await {
                Ok((stream, addr)) => self.handle(stream, addr).await,
                Err(err) => println!("Error: {}", err),
            }
        }
    }

    async fn handle(&mut self, stream: TcpStream, addr: SocketAddr) {
        let id = self.id;
        self.id.0 += 1;
        let server = self.server.clone();
        tokio::spawn(async move {
            let (rx, tx) = stream.into_split();
            let rx = FramedRead::new(rx, LengthDelimitedCodec::new());
            let mut rx = Framed::new(rx, Json::default());
            let tx = FramedWrite::new(tx, LengthDelimitedCodec::new());
            let tx = Framed::new(tx, Json::default());
            server.register_client(id, addr.ip(), ClientSender::start(tx));
            ClientReceiver::start(rx, id, server);
        });
    }
}

pub struct ClientListener;

impl ClientListener {
    pub fn start(port: u16, server: Server) {
        tokio::spawn(Actor::new(port, server).run());
    }
}
