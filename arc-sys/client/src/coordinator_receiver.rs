use shared::api::ClientAPI;
use shared::tcp;
use tokio_stream::StreamExt;

use crate::server::Server;

struct Actor {
    mailbox: tcp::Receiver<ClientAPI>,
    server: Server,
}

impl Actor {
    fn new(mailbox: tcp::Receiver<ClientAPI>, server: Server) -> Self {
        Self { mailbox, server }
    }

    async fn run(mut self) {
        while let Some(msg) = self.mailbox.next().await {
            match msg {
                Ok(msg) => self.handle(msg).await,
                Err(err) => println!("Error: {}", err),
            }
        }
    }

    async fn handle(&mut self, msg: ClientAPI) {
        match msg {
            ClientAPI::Query { source: _ } => unreachable!(),
            ClientAPI::QueryResponse { data } => self.server.query_response(data).await,
        }
    }
}

pub struct CoordinatorReceiver;

impl CoordinatorReceiver {
    pub fn start(rx: tcp::Receiver<ClientAPI>, server: Server) -> Self {
        tokio::spawn(Actor::new(rx, server).run());
        Self
    }
}
