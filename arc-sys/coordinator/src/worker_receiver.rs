use shared::api::CoordinatorAPI;
use shared::tcp;
use tokio_stream::StreamExt;

use crate::server::WorkerId;

use super::server::Server;

#[allow(unused)]
struct Actor {
    mailbox: tcp::Receiver<CoordinatorAPI>,
    id: WorkerId,
    server: Server,
}

pub struct WorkerReceiver;

impl Actor {
    fn new(mailbox: tcp::Receiver<CoordinatorAPI>, id: WorkerId, server: Server) -> Self {
        Self {
            mailbox,
            id,
            server,
        }
    }

    async fn run(mut self) {
        while let Some(msg) = self.mailbox.next().await {
            match msg {
                Ok(msg) => self.handle(msg).await,
                Err(err) => {
                    println!("Error: {}", err);
                }
            }
        }
    }

    async fn handle(&mut self, msg: CoordinatorAPI) {
        match msg {
            CoordinatorAPI::RegisterWorker { .. } => unreachable!(),
            CoordinatorAPI::RegisterClient => todo!(),
            CoordinatorAPI::Query { .. } => unreachable!(),
            CoordinatorAPI::Shutdown => todo!(),
        }
    }
}

impl WorkerReceiver {
    pub fn start(rx: tcp::Receiver<CoordinatorAPI>, id: WorkerId, server: Server) -> Self {
        tokio::spawn(Actor::new(rx, id, server).run());
        Self
    }
}
