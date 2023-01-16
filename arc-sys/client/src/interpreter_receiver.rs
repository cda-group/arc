use futures::StreamExt;
use shared::api::ClientAPI;
use shared::io;

use crate::server::Server;

struct Actor {
    mailbox: io::Receiver<ClientAPI>,
    server: Server,
}

impl Actor {
    fn new(mailbox: io::Receiver<ClientAPI>, server: Server) -> Self {
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

pub struct InterpreterReceiver;

impl InterpreterReceiver {
    pub fn start(rx: io::Receiver<ClientAPI>, server: Server) -> Self {
        tokio::spawn(Actor::new(rx, server).run());
        Self
    }
}
