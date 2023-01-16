use futures::SinkExt;
use shared::api::DataflowAPI;
use shared::io;
use tokio::sync::mpsc;

struct Actor {
    mailbox: mpsc::Receiver<Message>,
    tx: io::Sender<DataflowAPI>,
}

#[allow(unused)]
enum Message {
    Cast(DataflowAPI),
}

impl Actor {
    fn new(mailbox: mpsc::Receiver<Message>, tx: io::Sender<DataflowAPI>) -> Self {
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
                self.tx.send(msg).await.expect("Failed to send message");
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct DataflowSender(mpsc::Sender<Message>);

impl DataflowSender {
    pub fn start(tx_tcp: io::Sender<DataflowAPI>) -> Self {
        let (tx, rx) = mpsc::channel(8);
        tokio::spawn(Actor::new(rx, tx_tcp).run());
        Self(tx)
    }
}
