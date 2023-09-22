use api::Architecture;
use api::ClientId;
use api::Cluster;
use api::Query;
use api::Worker;
use api::WorkerId;
use halfbrown::HashMap;
use std::fmt::Debug;
use std::net::IpAddr;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::mpsc::Receiver;
use tokio::sync::mpsc::Sender;
use tokio::sync::oneshot;

use crate::rest_listener::RestListener;
use crate::worker_listener::TcpListener;
use crate::worker_tx::WorkerTx;
use crate::Args;

struct Actor {
    mailbox: Receiver<Message>,
    worker_txs: HashMap<WorkerId, WorkerTx>,
    cluster: Cluster,
    args: Args,
    server: Server,
    query_counter: usize,
}

// pub fn dummy_worker(target_triple: &str, num_cores: usize) -> Worker {
//     Worker {
//         id: WorkerId(0),
//         arch: Architecture {
//             target_triple: target_triple.to_owned(),
//             num_cpus: num_cores,
//         },
//         available_cpus: (0..num_cores).into_iter().collect(),
//         available_ports: (8000..9000).into_iter().collect(),
//         ip: IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
//     }
// }

#[derive(Clone)]
pub struct Server(Sender<Message>);

#[derive(Debug)]
enum Message {
    RegisterWorker {
        id: WorkerId,
        ip: IpAddr,
        arch: Architecture,
        tx: WorkerTx,
    },
    UnregisterWorker {
        id: WorkerId,
    },
    #[allow(unused)]
    BatchQuery {
        source: String,
        query: Query,
        tx: oneshot::Sender<String>,
    },
    StreamQuery {
        source: String,
        query: Query,
    },
    Shutdown,
}

impl Actor {
    fn new(mailbox: Receiver<Message>, server: Server, args: Args) -> Self {
        Self {
            mailbox,
            worker_txs: HashMap::new(),
            cluster: Cluster {
                workers: HashMap::new(),
                broker: args.broker,
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
                    id,
                    ip,
                    available_cpus: (0..arch.num_cpus).into_iter().collect(),
                    available_ports: (8000..9000).into_iter().collect(),
                    arch,
                };
                self.worker_txs.insert(id, tx);
                self.cluster.workers.insert(id, worker);
            }
            Message::UnregisterWorker { id } => {
                self.worker_txs.remove(&id);
                self.cluster.workers.remove(&id);
            }
            Message::BatchQuery { .. } => todo!(),
            Message::StreamQuery { source, query } => {
                let id = self.query_counter;
                self.query_counter += 1;
                tokio::runtime::Runtime::new()
                    .expect("Failed to create tokio runtime")
                    .block_on(self.stream_query(id, &source, query))
            }
            Message::Shutdown => todo!(),
        }
    }

    pub async fn stream_query(&mut self, id: usize, source: &str, query: Query) {
        let name = Arc::new(format!("package{id}"));
        let graph = query_compiler::compile(&name, source, query, &mut self.cluster);
        for (worker_ids, binary_path) in graph.deployment {
            let binary = Arc::new(std::fs::read(binary_path).expect("Failed to read binary"));
            for worker_id in worker_ids {
                let worker = self.worker_txs.get(&worker_id).unwrap();
                worker.execute(name.clone(), binary.clone()).await;
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
        tx: WorkerTx,
    ) {
        self.0
            .send(Message::RegisterWorker { id, ip, arch, tx })
            .await
            .expect("Failed to send");
    }

    pub async fn batch_query(&self, source: String, config: Query) -> String {
        let (tx, rx) = oneshot::channel();
        self.0
            .send(Message::BatchQuery {
                source,
                query: config,
                tx,
            })
            .await
            .expect("Failed to send");
        rx.await.expect("Server is dead")
    }

    pub async fn stream_query(&self, source: String, config: Query) {
        self.0
            .send(Message::StreamQuery {
                source,
                query: config,
            })
            .await
            .expect("Failed to send");
    }

    pub async fn shutdown(&self) {
        self.0
            .send(Message::Shutdown)
            .await
            .expect("Failed to forward");
    }

    pub async fn register_client(&self, _id: ClientId, _ip: IpAddr) {
        todo!()
    }
}
