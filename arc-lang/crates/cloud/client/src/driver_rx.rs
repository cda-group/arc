use futures::StreamExt;
use api::ClientAPI;
use io::term;

use crate::server::Server;

struct Actor {
    mailbox: term::Rx<ClientAPI>,
    server: Server,
}

impl Actor {
    fn new(mailbox: term::Rx<ClientAPI>, server: Server) -> Self {
        Self { mailbox, server }
    }

    async fn run(mut self) {
        while let Some(msg) = self.mailbox.next().await {
            match msg {
                Ok(msg) => self.handle(msg).await,
                Err(_) => todo!(),
            }
        }
    }

    async fn handle(&mut self, msg: ClientAPI) {
        match msg {
            ClientAPI::Query { source } => self.server.query(source).await,
            ClientAPI::QueryResponse { .. } => unreachable!(),
        }
    }
}

pub struct DriverRx;

impl DriverRx {
    pub fn start(rx: term::Rx<ClientAPI>, server: Server) -> Self {
        tokio::spawn(Actor::new(rx, server).run());
        Self
    }
}
