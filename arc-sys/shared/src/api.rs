use halfbrown::HashMap;
use indexmap::IndexMap;
use std::net::SocketAddr;
use std::ops::Range;
use std::sync::Arc;

use serde::Deserialize;
use serde::Serialize;

/// Messages that can be sent to a worker.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum WorkerAPI {
    /// Execute a program on the worker.
    Execute {
        name: Arc<String>,
        binary: Arc<Vec<u8>>,
    },
    /// Shutdown the worker.
    Shutdown,
}

/// Messages that can be sent to a coordinator.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum CoordinatorAPI {
    /// Register a new worker.
    RegisterWorker { arch: Architecture },
    /// Register a new client.
    RegisterClient,
    /// Post a new query.
    Query { source: String, config: QueryConfig },
    /// Shutdown the coordinator (and consequently the whole system).
    Shutdown,
}

/// Messages that can be sent to a client.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum ClientAPI {
    /// Request to execute query.
    Query { source: String },
    /// Result of the query.
    QueryResponse { data: String },
}

/// Responses that can be sent to an interpreter.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum InterpreterAPI {
    /// Begin interpreting file.
    InterpretFile { path: String },
    /// Result of a query.
    QueryResponse { data: String },
}

/// The config of a query
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct QueryConfig {
    pub parallelism: usize,
    pub state_backend: StateBackend,
}

/// The state backend to use for a query.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum StateBackend {
    Sled,
    TiKV,
}

/// Requests that can be sent to a pipeline.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum DataflowAPI {
    /// Shutdown the pipeline.
    Shutdown,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Architecture {
    pub target_triple: String,
    pub num_cpus: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: IndexMap<String, Node>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Node {
    Filter {
        input: String, // Input stream variable name
        fun: String,   // Predicate function name
    },
    Map {
        input: String, // Input stream variable name
        fun: String,   // Mapping function name
    },
    Source {
        key_type: String,    // Key type name
        data_type: String,   // Data type name
        topic: String,       // Kafka topic
        num_partitions: u32, // Partitions to consume from.
    },
    Sink {
        input: String, // Input stream variable name
        topic: String, // Kafka topic
        fun: String,   // Key extractor function name
    },
    Group {
        input: String, // Input stream
        fun: String,   // Key extractor
    },
    Union {
        input0: String, // Input stream 0 variable name
        input1: String, // Input stream 1 variable name
    },
    Window {
        input: String, // Input stream variable name
    },
}
