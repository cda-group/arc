pub mod lowering1;
pub mod lowering2;
pub mod lowering3;
pub mod lowering4;
pub mod lowering5;
pub mod lowering6;
pub mod lowering7;

use shared::api::QueryConfig;

use crate::server::ServerConfig;

use self::lowering7::Graph7;

pub async fn compile(
    name: &str,
    source: &str,
    query_config: &QueryConfig,
    server_config: &mut ServerConfig,
) -> Graph7 {
    // Compile MLIR source code to Rust and a dataflow graph.
    let graph1 = lowering1::lower(source);
    // Compile logical operators into physical operators.
    let graph2 = lowering2::lower(graph1);
    // Topologically sort nodes.
    let graph3 = lowering3::lower(graph2);
    // Extract pipelines.
    let graph4 = lowering4::lower(graph3);
    // Allocate cpus and sockets to pipelines.
    let graph5 = lowering5::lower(graph4, server_config, query_config);
    // Compile pipelines to source code.
    let graph6 = lowering6::lower(graph5, server_config);
    // Compile source code to binaries.
    let graph7 = lowering7::lower(&name, graph6, server_config).await;
    graph7
}

#[test]
fn test_compile() {
    use crate::compiler::lowering1::Graph1;
    use crate::compiler::lowering6::Graph6;
    use crate::server::Worker;
    use crate::server::WorkerId;
    use shared::api::Node;
    use shared::api::StateBackend;
    use std::net::ToSocketAddrs;
    fn lower(
        graph1: Graph1,
        query_config: &QueryConfig,
        server_config: &mut ServerConfig,
    ) -> Graph6 {
        // Compile logical operators into physical operators.
        let graph2 = lowering2::lower(graph1);
        // Topologically sort nodes.
        let graph3 = lowering3::lower(graph2);
        // Extract pipelines.
        let graph4 = lowering4::lower(graph3);
        // Allocate cpus and sockets to pipelines.
        let graph5 = lowering5::lower(graph4, server_config, query_config);
        // Compile pipelines to source code.
        let graph6 = lowering6::lower(graph5, server_config);
        graph6
    }
    lower(
        Graph1 {
            code: quote::quote!(
                fn filter_udf(x: i32) -> bool {
                    x > 0
                }
                fn map_udf(x: i32) -> i32 {
                    x + 1
                }
                fn key_udf(x: i32) -> i32 {
                    x % 2
                }
            )
            .to_string(),
            nodes: vec![
                (
                    "x0".to_string(),
                    Node::Source {
                        key_type: "i32".to_string(),
                        data_type: "i32".to_string(),
                        topic: "my-topic-1".to_string(),
                        num_partitions: 10,
                    },
                ),
                (
                    "x1".to_string(),
                    Node::Filter {
                        input: "x0".to_string(),
                        fun: "filter_udf".to_string(),
                    },
                ),
                (
                    "x2".to_string(),
                    Node::Group {
                        input: "x1".to_string(),
                        fun: "key_udf".to_string(),
                    },
                ),
                (
                    "x3".to_string(),
                    Node::Map {
                        input: "x2".to_string(),
                        fun: "map_udf".to_string(),
                    },
                ),
                (
                    "x4".to_string(),
                    Node::Sink {
                        input: "x3".to_string(),
                        topic: "my-topic-2".to_string(),
                        fun: "key_udf".to_string(),
                    },
                ),
            ]
            .into_iter()
            .collect(),
        },
        &QueryConfig {
            parallelism: 2,
            state_backend: StateBackend::Sled,
        },
        &mut ServerConfig {
            broker: "localhost:9092".to_socket_addrs().unwrap().next().unwrap(),
            workers: vec![
                (WorkerId(0), Worker::dummy("aarch64-unknown-linux-gnu", 2)),
                (WorkerId(1), Worker::dummy("aarch64-unknown-linux-gnu", 2)),
                (WorkerId(2), Worker::dummy("aarch64-unknown-linux-gnu", 2)),
                (WorkerId(3), Worker::dummy("aarch64-unknown-linux-gnu", 2)),
            ]
            .into_iter()
            .collect(),
        },
    );
}
