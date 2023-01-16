use std::collections::HashMap;
use std::env::temp_dir;
use std::fs::File;
use std::io::Write;
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio_serde::formats::Json;
use tokio_serde::Framed;
use tokio_util::codec::FramedRead;
use tokio_util::codec::FramedWrite;
use tokio_util::codec::LengthDelimitedCodec;

use crate::coordinator_connector::CoordinatorConnector;
use crate::coordinator_sender::CoordinatorSender;
use crate::dataflow_receiver::DataflowReceiver;
use crate::dataflow_sender::DataflowSender;
use crate::Args;

struct Actor {
    mailbox: mpsc::Receiver<Message>,
    tx: Option<CoordinatorSender>,
    txs: HashMap<u64, DataflowSender>,
    server: Server,
    args: Args,
    id: u64,
}

#[derive(Clone)]
pub struct Server(mpsc::Sender<Message>);

#[derive(Debug)]
enum Message {
    Connect { tx: CoordinatorSender },
    Execute { name: String, binary: Vec<u8> },
    Shutdown,
}

impl Actor {
    fn new(mailbox: mpsc::Receiver<Message>, server: Server, args: Args) -> Self {
        Self {
            mailbox,
            tx: None,
            txs: HashMap::new(),
            server,
            id: 0,
            args,
        }
    }

    async fn run(mut self) {
        CoordinatorConnector::start(self.args.coordinator, self.server.clone());
        while let Some(msg) = self.mailbox.recv().await {
            self.handle(msg);
        }
    }

    fn handle(&mut self, msg: Message) {
        match msg {
            Message::Connect { tx } => {
                self.tx = Some(tx);
            }
            Message::Execute { name, binary } => {
                let path = temp_dir().join("arc-lang").join(name);

                let mut file = File::create(&path).unwrap();
                file.write_all(&binary).unwrap();
                file.flush().unwrap();
                drop(file);

                let mut child = Command::new(path).spawn().expect("Spawning failed");

                let rx = child.stdout.take().expect("Failed to get stdin");
                let tx = child.stdin.take().expect("Failed to get stdin");

                let rx = FramedRead::new(rx, LengthDelimitedCodec::new());
                let rx = Framed::new(rx, Json::default());
                let tx = FramedWrite::new(tx, LengthDelimitedCodec::new());
                let tx = Framed::new(tx, Json::default());

                self.txs.insert(self.id, DataflowSender::start(tx));
                DataflowReceiver::start(rx, self.id, self.server.clone());
            }
            Message::Shutdown => self.mailbox.close(),
        }
    }
}

impl Server {
    pub async fn start(args: Args) {
        let (tx, rx) = mpsc::channel(100);
        let server = Self(tx);
        tokio::spawn(Actor::new(rx, server, args).run())
            .await
            .expect("Failed to spawn");
    }

    pub async fn connect(&self, tx: CoordinatorSender) {
        self.0.send(Message::Connect { tx }).await.unwrap();
    }

    pub async fn execute(&self, name: String, binary: Vec<u8>) {
        self.0
            .send(Message::Execute { name, binary })
            .await
            .expect("failed to forward");
    }

    pub async fn shutdown(&self) {
        self.0.send(Message::Shutdown).await.unwrap();
    }
}
