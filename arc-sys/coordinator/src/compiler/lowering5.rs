use halfbrown::HashMap;
use std::net::IpAddr;
use std::net::SocketAddr;


use shared::api::QueryConfig;

use crate::server::ServerConfig;
use crate::server::WorkerId;

use super::lowering2::Node2;
use super::lowering2::NodeId;
use super::lowering3::PipelineId;
use super::lowering4::Graph4;
use super::lowering4::Pipeline3;

#[derive(Debug)]
pub struct Graph5 {
    pub code: String,
    pub pipelines: HashMap<PipelineId, Pipeline3>,
    pub shards: Vec<Shard4>,
    // All sinks need to connect to the same sources.
    pub sink_ports: HashMap<NodeId, Vec<SocketAddr>>,
    pub parallelism: usize,
}

#[derive(Debug)]
pub struct Shard4 {
    pub worker_id: WorkerId,
    pub pipeline_id: PipelineId,
    pub cpu: usize,
    pub source_ip: IpAddr,
    pub source_ports: HashMap<NodeId, u16>,
}

/// * Each pipeline is sharded by a parallelism factor, currently specified by the query.
/// * Each shard is mapped to one thread which executes on one CPU.
/// * Multiple threads can be collocated in the same OS-process.
/// * Multiple OS-processes could be collocated on the same worker.
/// * Each machine has one worker and potentially multiple cores.
/// * Each Source and Sink operation needs to be connected to its system (e.g., Kafka).
/// * Each ShuffleSink must open a connection to its respective ShuffleSource operator in the next shard.
pub fn lower(
    graph: Graph4,
    server_config: &mut ServerConfig,
    query_config: &QueryConfig,
) -> Graph5 {
    let mut instances = Vec::new();
    let mut sinks = HashMap::new();
    let mut workers = server_config.workers.iter_mut();
    let (mut worker_id, mut worker) = workers.next().unwrap();
    for (pipeline_id, pipeline) in &graph.pipelines {
        // For each physical ShuffleSource we need to allocate one port.
        let sources: Vec<_> = pipeline
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                if let Node2::ShuffleSource { input, .. } = node {
                    sinks.insert(*input, Vec::new());
                    Some((*id, input))
                } else {
                    None
                }
            })
            .collect();
        for _ in 0..query_config.parallelism {
            // Collocate the same pipeline on the same worker if possible.
            while worker.available_cpus.is_empty() {
                (worker_id, worker) = workers.next().expect("Insufficient cores available");
            }
            let cpu = worker.available_cpus.pop_first().unwrap();
            let ports = sources
                .iter()
                .map(|(id, input)| {
                    if let Some(port) = worker.available_ports.pop_first() {
                        sinks
                            .get_mut(*input)
                            .unwrap()
                            .push(SocketAddr::new(worker.ip, port));
                        (*id, port)
                    } else {
                        panic!("Insufficient ports available");
                    }
                })
                .collect();
            instances.push(Shard4 {
                worker_id: *worker_id,
                pipeline_id: *pipeline_id,
                source_ip: worker.ip,
                cpu,
                source_ports: ports,
            });
        }
    }
    Graph5 {
        code: graph.code,
        sink_ports: sinks,
        pipelines: graph.pipelines,
        shards: instances,
        parallelism: query_config.parallelism,
    }
}
