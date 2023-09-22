#![allow(unused)]

use crate::coordinator_connector::CoordinatorConnector;
use crate::coordinator_tx::CoordinatorTx;
use crate::driver_rx::DriverRx;
use crate::driver_tx::DriverTx;
use crate::Args;
use tokio::process::Command;
use tokio::sync::mpsc::Receiver;
use tokio::sync::mpsc::Sender;
use tokio_serde::formats::Json;
use tokio_serde::Framed;
use tokio_util::codec::FramedRead;
use tokio_util::codec::FramedWrite;
use tokio_util::codec::LengthDelimitedCodec;

struct Actor {
    mailbox: Receiver<Message>,
    tx: Option<CoordinatorTx>,
    server: Server,
    args: Args,
}

#[derive(Debug)]
enum Message {
    Query { source: String },
    QueryResponse { data: String },
    Connect { tx: CoordinatorTx },
    Execute { source: String },
    Shutdown,
}

#[derive(Clone)]
pub struct Server(Sender<Message>);

impl Actor {
    fn new(mailbox: Receiver<Message>, server: Server, args: Args) -> Self {
        Self {
            mailbox,
            tx: None,
            server,
            args,
        }
    }

    async fn run(mut self) {
        CoordinatorConnector::start(self.args.coordinator, self.server.clone());

        todo!()
        // let tx = DriverSender::start(tx);
        // DriverReceiver::start(rx, self.server.clone());
        //
        // while let Some(msg) = self.mailbox.recv().await {
        //     self.handle(msg);
        // }
    }

    fn handle(&mut self, msg: Message) {
        match msg {
            Message::Query { source } => {
                println!("Query: {}", source);
            }
            Message::Shutdown => {
                println!("Shutdown");
            }
            Message::Connect { tx } => todo!(),
            Message::Execute { source } => todo!(),
            Message::QueryResponse { data } => todo!(),
        }
    }
}

impl Server {
    pub async fn new(args: Args) {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let server = Self(tx);
        tokio::spawn(Actor::new(rx, server, args).run())
            .await
            .expect("Failed to spawn");
    }

    pub async fn connect(&self, tx: CoordinatorTx) {
        self.0.send(Message::Connect { tx }).await.unwrap();
    }

    pub async fn query(&self, source: String) {
        self.0
            .send(Message::Execute { source })
            .await
            .expect("failed to forward");
    }

    pub async fn shutdown(&self) {
        self.0.send(Message::Shutdown).await.unwrap();
    }

    pub(crate) async fn query_response(&self, data: String) {
        self.0.send(Message::QueryResponse { data }).await.unwrap();
    }
}
