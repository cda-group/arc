use futures::SinkExt;
use api::WorkerAPI;
use io::tcp;
use std::sync::Arc;
use tokio::sync::mpsc;

struct Actor {
    mailbox: mpsc::Receiver<Message>,
    tx: tcp::Tx<WorkerAPI>,
}

#[derive(Debug)]
enum Message {
    Cast(WorkerAPI),
}

#[derive(Debug, Clone)]
pub struct WorkerTx(mpsc::Sender<Message>);

impl Actor {
    fn new(mailbox: mpsc::Receiver<Message>, tx: tcp::Tx<WorkerAPI>) -> Self {
        Self { mailbox, tx }
    }

    async fn run(mut self) {
        while let Some(msg) = self.mailbox.recv().await {
            self.handle(msg).await;
        }
    }

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Cast(msg) => {
                self.tx.send(msg).await.unwrap();
            }
        }
    }
}

impl WorkerTx {
    pub fn start(tx_tcp: tcp::Tx<WorkerAPI>) -> Self {
        let (tx, rx) = mpsc::channel(100);
        tokio::spawn(Actor::new(rx, tx_tcp).run());
        Self(tx)
    }

    pub fn dummy() -> Self {
        let (tx, _) = mpsc::channel(100);
        Self(tx)
    }

    pub async fn execute(&self, name: Arc<String>, binary: Arc<Vec<u8>>) {
        self.0
            .send(Message::Cast(WorkerAPI::Execute { name, binary }))
            .await
            .expect("Failed to send");
    }
}
