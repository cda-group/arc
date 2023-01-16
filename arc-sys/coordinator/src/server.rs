use shared::api::Architecture;
use shared::api::QueryConfig;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::fmt::Debug;
use std::net::IpAddr;
use std::net::Ipv4Addr;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::mpsc::Receiver;
use tokio::sync::mpsc::Sender;
use tokio::sync::oneshot;

use crate::rest_listener::RestListener;
use crate::worker_listener::TcpListener;
use crate::worker_sender::WorkerSender;
use crate::Args;

struct Actor {
    mailbox: Receiver<Message>,
    config: ServerConfig,
    args: Args,
    server: Server,
    query_counter: usize,
}

pub struct Worker {
    pub tx: WorkerSender,
    pub arch: Architecture,
    pub available_cpus: BTreeSet<usize>,
    pub available_ports: BTreeSet<u16>,
    pub ip: IpAddr,
}

impl Worker {
    pub fn dummy(target_triple: &str, num_cores: usize) -> Self {
        Self {
            tx: WorkerSender::dummy(),
            arch: Architecture {
                target_triple: target_triple.to_owned(),
                num_cpus: num_cores,
            },
            available_cpus: (0..num_cores).into_iter().collect(),
            available_ports: (8000..9000).into_iter().collect(),
            ip: IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
        }
    }
}

pub struct ServerConfig {
    pub broker: SocketAddr,
    pub workers: HashMap<WorkerId, Worker>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct WorkerId(pub u32);

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ClientId(pub u32);

#[derive(Clone)]
pub struct Server(Sender<Message>);

#[derive(Debug)]
enum Message {
    RegisterWorker {
        id: WorkerId,
        ip: IpAddr,
        arch: Architecture,
        tx: WorkerSender,
    },
    UnregisterWorker {
        id: WorkerId,
    },
    #[allow(unused)]
    BatchQuery {
        source: String,
        config: QueryConfig,
        tx: oneshot::Sender<String>,
    },
    StreamQuery {
        source: String,
        config: QueryConfig,
    },
    Shutdown,
}

impl Actor {
    fn new(mailbox: Receiver<Message>, server: Server, args: Args) -> Self {
        Self {
            mailbox,
            config: ServerConfig {
                broker: args.broker,
                workers: HashMap::new(),
            },
            args,
            server,
            query_counter: 0,
        }
    }

    async fn run(mut self) {
        TcpListener::start(self.args.tcp_listener_port, self.server.clone());
        RestListener::start(self.args.tcp_listener_port, self.server.clone());
        while let Some(msg) = self.mailbox.recv().await {
            self.handle(msg);
        }
    }

    fn handle(&mut self, msg: Message) {
        match msg {
            Message::RegisterWorker { id, ip, arch, tx } => {
                let worker = Worker {
                    tx,
                    ip,
                    available_cpus: (0..arch.num_cpus).into_iter().collect(),
                    available_ports: (8000..9000).into_iter().collect(),
                    arch,
                };
                self.config.workers.insert(id, worker);
            }
            Message::UnregisterWorker { id } => {
                self.config.workers.remove(&id);
            }
            Message::BatchQuery { .. } => todo!(),
            Message::StreamQuery { source, config } => {
                let id = self.query_counter;
                self.query_counter += 1;
                tokio::runtime::Runtime::new()
                    .expect("Failed to create tokio runtime")
                    .block_on(self.stream_query(id, &source, &config))
            }
            Message::Shutdown => todo!(),
        }
    }

    pub async fn stream_query(&mut self, id: usize, source: &str, config: &QueryConfig) {
        let name = Arc::new(format!("package{id}"));
        let graph = crate::compiler::compile(&name, source, config, &mut self.config).await;
        for (worker_ids, binary_path) in graph.deployment {
            let binary = Arc::new(std::fs::read(binary_path).expect("Failed to read binary"));
            for worker_id in worker_ids {
                let worker = self.config.workers.get(&worker_id).unwrap();
                worker.tx.execute(name.clone(), binary.clone()).await;
            }
        }
    }
}

impl Server {
    pub async fn start(args: Args) {
        let (tx, rx) = mpsc::channel(100);
        tokio::spawn(Actor::new(rx, Self(tx), args).run())
            .await
            .expect("Failed to spawn actor");
    }

    pub async fn unregister_worker(&self, id: WorkerId) {
        self.0
            .send(Message::UnregisterWorker { id })
            .await
            .expect("Failed to send");
    }

    pub async fn register_worker(
        &self,
        id: WorkerId,
        ip: IpAddr,
        arch: Architecture,
        tx: WorkerSender,
    ) {
        self.0
            .send(Message::RegisterWorker { id, ip, arch, tx })
            .await
            .expect("Failed to send");
    }

    pub async fn batch_query(&self, source: String, config: QueryConfig) -> String {
        let (tx, rx) = oneshot::channel();
        self.0
            .send(Message::BatchQuery { source, config, tx })
            .await
            .expect("Failed to send");
        rx.await.expect("Server is dead")
    }

    pub async fn stream_query(&self, source: String, config: QueryConfig) {
        self.0
            .send(Message::StreamQuery { source, config })
            .await
            .expect("Failed to send");
    }

    pub async fn shutdown(&self) {
        self.0
            .send(Message::Shutdown)
            .await
            .expect("Failed to forward");
    }
}
