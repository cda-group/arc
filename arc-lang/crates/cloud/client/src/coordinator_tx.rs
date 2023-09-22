use futures::SinkExt;
use api::CoordinatorAPI;
use io::tcp;
use tokio::sync::mpsc;

struct Actor {
    mailbox: mpsc::Receiver<Message>,
    tx: tcp::Tx<CoordinatorAPI>,
}

#[allow(unused)]
enum Message {
    Cast(CoordinatorAPI),
}

impl Actor {
    fn new(mailbox: mpsc::Receiver<Message>, tx: tcp::Tx<CoordinatorAPI>) -> Self {
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
pub struct CoordinatorTx(mpsc::Sender<Message>);

impl CoordinatorTx {
    pub fn start(tx_tcp: tcp::Tx<CoordinatorAPI>) -> Self {
        let (tx, rx) = mpsc::channel(10);
        tokio::spawn(Actor::new(rx, tx_tcp).run());
        Self(tx)
    }
}
