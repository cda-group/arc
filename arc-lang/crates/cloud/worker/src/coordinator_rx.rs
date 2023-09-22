use api::WorkerAPI;
use io::tcp;
use std::sync::Arc;
use tokio_stream::StreamExt;

use crate::server::Server;

struct Actor {
    mailbox: tcp::Rx<WorkerAPI>,
    server: Server,
}

impl Actor {
    fn new(mailbox: tcp::Rx<WorkerAPI>, server: Server) -> Self {
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

    async fn handle(&mut self, msg: WorkerAPI) {
        match msg {
            WorkerAPI::Execute { name, binary } => {
                self.server
                    .execute(
                        Arc::try_unwrap(name).unwrap(),
                        Arc::try_unwrap(binary).unwrap(),
                    )
                    .await
            }
            WorkerAPI::Shutdown => self.server.shutdown().await,
        }
    }
}

pub struct CoordinatorRx;

impl CoordinatorRx {
    pub fn start(rx: tcp::Rx<WorkerAPI>, server: Server) -> Self {
        tokio::spawn(Actor::new(rx, server).run());
        Self
    }
}
