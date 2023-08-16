use api::CoordinatorAPI;
use io::tcp;
use tokio_stream::StreamExt;

use api::WorkerId;

use crate::server::Server;

#[allow(unused)]
struct Actor {
    mailbox: tcp::Rx<CoordinatorAPI>,
    id: WorkerId,
    server: Server,
}

pub struct WorkerRx;

impl Actor {
    fn new(mailbox: tcp::Rx<CoordinatorAPI>, id: WorkerId, server: Server) -> Self {
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

impl WorkerRx {
    pub fn start(rx: tcp::Rx<CoordinatorAPI>, id: WorkerId, server: Server) -> Self {
        tokio::spawn(Actor::new(rx, id, server).run());
        Self
    }
}
