use futures::SinkExt;
use shared::api::CoordinatorAPI;
use shared::tcp;
use tokio::sync::mpsc;

struct Actor {
    mailbox: mpsc::Receiver<Message>,
    tx: tcp::Sender<CoordinatorAPI>,
}

#[allow(unused)]
enum Message {
    Cast(CoordinatorAPI),
}

impl Actor {
    fn new(mailbox: mpsc::Receiver<Message>, tx: tcp::Sender<CoordinatorAPI>) -> Self {
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
                self.tx.send(msg).await.expect("failed sending message");
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct CoordinatorSender(mpsc::Sender<Message>);

impl CoordinatorSender {
    pub fn start(tx_tcp: tcp::Sender<CoordinatorAPI>) -> Self {
        let (tx, rx) = mpsc::channel(10);
        tokio::spawn(Actor::new(rx, tx_tcp).run());
        Self(tx)
    }
}
