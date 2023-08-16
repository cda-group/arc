use api::RuntimeAPI;
use futures::SinkExt;
use io::term;
use tokio::sync::mpsc;

struct Actor {
    mailbox: mpsc::Receiver<Message>,
    tx: term::Tx<RuntimeAPI>,
}

#[allow(unused)]
enum Message {
    Cast(RuntimeAPI),
}

impl Actor {
    fn new(mailbox: mpsc::Receiver<Message>, tx: term::Tx<RuntimeAPI>) -> Self {
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
pub struct RuntimeTx(mpsc::Sender<Message>);

impl RuntimeTx {
    pub fn start(tx_tcp: term::Tx<RuntimeAPI>) -> Self {
        let (tx, rx) = mpsc::channel(8);
        tokio::spawn(Actor::new(rx, tx_tcp).run());
        Self(tx)
    }
}
