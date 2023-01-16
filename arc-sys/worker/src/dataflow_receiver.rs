use futures::StreamExt;
use shared::api::DataflowAPI;
use shared::io;

use crate::server::Server;

#[allow(unused)]
struct Actor {
    mailbox: io::Receiver<DataflowAPI>,
    id: u64,
    server: Server,
}

impl Actor {
    fn new(mailbox: io::Receiver<DataflowAPI>, id: u64, server: Server) -> Self {
        Self { mailbox, id, server }
    }

    async fn run(mut self) {
        while let Some(msg) = self.mailbox.next().await {
            match msg {
                Ok(msg) => self.handle(msg).await,
                Err(_) => todo!(),
            }
        }
    }

    async fn handle(&mut self, msg: DataflowAPI) {
        match msg {
            DataflowAPI::Shutdown => {}
        }
    }
}

pub struct DataflowReceiver;

impl DataflowReceiver {
    pub fn start(rx: io::Receiver<DataflowAPI>, id: u64, server: Server) -> Self {
        tokio::spawn(Actor::new(rx, id, server).run());
        Self
    }
}
