use std::collections::hash_map::Entry;
use std::collections::HashMap;
use shared::api::Node;

use super::lowering1::Graph1;

#[derive(Clone, Debug)]
pub struct Graph2 {
    pub code: String,
    pub nodes: HashMap<NodeId, Node2>,
}

#[derive(Clone, Copy, Debug, Ord, PartialOrd, Eq, Hash, PartialEq)]
pub struct NodeId(pub u32);

#[derive(Clone, Debug)]
pub enum Node2 {
    Filter {
        input: NodeId,
        fun: String,
    },
    Map {
        input: NodeId,
        fun: String,
    },
    KafkaSource {
        key_type: String,
        data_type: String,
        topic: String,
        num_partitions: u32,
    },
    KafkaSink {
        input: NodeId,
        topic: String,
        fun: String,
    },
    ShuffleSink {
        input: NodeId,
        fun: String,
    },
    ShuffleSource {
        input: NodeId,
        fun: String,
    },
    Union {
        input0: NodeId,
        input1: NodeId,
    },
    Window {
        input: NodeId,
    },
}

struct IdIntern {
    map: HashMap<String, NodeId>,
    counter: NodeId,
}

impl IdIntern {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            counter: NodeId(0),
        }
    }
    fn intern(&mut self, name: String) -> NodeId {
        match self.map.entry(name) {
            Entry::Occupied(v) => *v.get(),
            Entry::Vacant(v) => {
                let id = self.counter;
                v.insert(id);
                self.counter.0 += 1;
                id
            }
        }
    }
    fn create(&mut self) -> NodeId {
        let id = self.counter;
        self.counter.0 += 1;
        id
    }
}

// 1. Split Group into ShuffleSink and ShuffleSource
// 2. Replace symbolic identifiers (String) with numerical ones (NodeId)
// NOTE: We currently assume that the graph is already topologically sorted
pub fn lower(graph: Graph1) -> Graph2 {
    let mut nodes = HashMap::new();
    let mut ids = IdIntern::new();
    for (name, node) in graph.nodes.into_iter() {
        match node {
            Node::Map { input, fun } => {
                let id = ids.intern(name);
                let input = ids.intern(input);
                nodes.insert(id, Node2::Map { input, fun });
            }
            Node::Sink { input, topic, fun } => {
                let id = ids.intern(name);
                let input = ids.intern(input);
                nodes.insert(id, Node2::KafkaSink { input, topic, fun });
            }
            Node::Union { input0, input1 } => {
                let id = ids.intern(name);
                let input0 = ids.intern(input0);
                let input1 = ids.intern(input1);
                nodes.insert(id, Node2::Union { input0, input1 });
            }
            Node::Group { input, fun } => {
                let source_id = ids.create();
                let sink_id = ids.intern(name);
                let input = ids.intern(input);
                nodes.insert(
                    source_id,
                    Node2::ShuffleSink {
                        input,
                        fun: fun.clone(),
                    },
                );
                nodes.insert(
                    sink_id,
                    Node2::ShuffleSource {
                        input: source_id,
                        fun,
                    },
                );
            }
            Node::Filter { input, fun } => {
                let id = ids.intern(name);
                let input = ids.intern(input);
                nodes.insert(id, Node2::Filter { input, fun });
            }
            Node::Source {
                key_type,
                data_type,
                topic,
                num_partitions,
            } => {
                let id = ids.intern(name);
                nodes.insert(
                    id,
                    Node2::KafkaSource {
                        key_type,
                        data_type,
                        topic,
                        num_partitions,
                    },
                );
            }
            Node::Window { input } => {
                let id = ids.intern(name);
                let input = ids.intern(input);
                nodes.insert(id, Node2::Window { input });
            }
        }
    }
    Graph2 {
        code: graph.code,
        nodes,
    }
}
