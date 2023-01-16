use futures::SinkExt;
use shared::api::InterpreterAPI;
use shared::io;
use tokio::sync::mpsc;

struct Actor {
    mailbox: mpsc::Receiver<Message>,
    tx: io::Sender<InterpreterAPI>,
}

#[allow(unused)]
#[derive(Debug)]
enum Message {
    Cast(InterpreterAPI),
}

impl Actor {
    fn new(mailbox: mpsc::Receiver<Message>, tx: io::Sender<InterpreterAPI>) -> Self {
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
pub struct InterpreterSender(mpsc::Sender<Message>);

impl InterpreterSender {
    pub fn start(tx_tcp: io::Sender<InterpreterAPI>) -> Self {
        let (tx, rx) = mpsc::channel(8);
        tokio::spawn(Actor::new(rx, tx_tcp).run());
        Self(tx)
    }

    pub async fn query_response(&self, data: String) {
        self.0
            .send(Message::Cast(InterpreterAPI::QueryResponse { data }))
            .await
            .expect("Failed to send message");
    }
}
