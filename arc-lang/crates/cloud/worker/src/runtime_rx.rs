use api::RuntimeAPI;
use futures::StreamExt;
use io::term;

use crate::server::Server;

#[allow(unused)]
struct Actor {
    mailbox: term::Rx<RuntimeAPI>,
    id: u64,
    server: Server,
}

impl Actor {
    fn new(mailbox: term::Rx<RuntimeAPI>, id: u64, server: Server) -> Self {
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
                Err(_) => todo!(),
            }
        }
    }

    async fn handle(&mut self, msg: RuntimeAPI) {
        match msg {
            RuntimeAPI::Shutdown => {}
        }
    }
}

pub struct RuntimeRx;

impl RuntimeRx {
    pub fn start(rx: term::Rx<RuntimeAPI>, id: u64, server: Server) -> Self {
        tokio::spawn(Actor::new(rx, id, server).run());
        Self
    }
}
