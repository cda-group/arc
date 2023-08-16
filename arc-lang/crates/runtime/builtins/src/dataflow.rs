#[derive(Debug)]
pub struct Dataflow {
    pub tx: tokio::sync::broadcast::Sender<()>,
}

impl Dataflow {
    pub fn new(tx: tokio::sync::broadcast::Sender<()>) -> Self {
        Self { tx }
    }
    pub fn run(self) {
        self.tx.send(()).unwrap();
    }
}
