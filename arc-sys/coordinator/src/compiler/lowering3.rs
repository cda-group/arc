use super::lowering2::Graph2;
use super::lowering2::Node2;
use super::lowering2::NodeId;

use halfbrown::HashMap;

#[derive(Clone, Debug)]
pub struct Graph3 {
    pub code: String,
    pub pipelines: HashMap<PipelineId, Pipeline2>,
}

#[derive(Clone, Copy, Debug, Ord, PartialOrd, Eq, Hash, PartialEq)]
pub struct PipelineId(pub u32);

#[derive(Clone, Debug, Default)]
pub struct Pipeline2 {
    pub nodes: HashMap<NodeId, Node2>,
}

mod datalog {

    use crate::compiler::lowering2::Graph2;
    use crate::compiler::lowering2::NodeId;

    crepe::crepe! {
        @input
        struct Edge(NodeId, NodeId);

        @input
        struct Node(NodeId, bool);

        struct Reachable(NodeId, NodeId);

        Reachable(x, y) <- Edge(x, y);
        Reachable(x, z) <- Reachable(x, y), Edge(y, z);

        struct PipelineBreaker(NodeId);

        PipelineBreaker(x) <- Node(x, true);

        @output
        #[derive(Debug, Ord, PartialOrd)]
        struct Pipeline(NodeId, NodeId);

        Pipeline(x, x) <- PipelineBreaker(x);
        Pipeline(x, y) <- Reachable(x, y), PipelineBreaker(y);
    }

    pub fn run(graph1: &Graph2) -> impl Iterator<Item = (NodeId, NodeId)> {
        let nodes = graph1
            .nodes
            .iter()
            .map(|(id, node)| Node(*id, node.is_sink()));

        let edges = graph1
            .nodes
            .iter()
            .flat_map(|(id, node)| node.data_inputs().map(|input| Edge(*input, *id)));

        let mut runtime = Crepe::new();
        runtime.extend(edges);
        runtime.extend(nodes);
        let (pairs,) = runtime.run();
        pairs.into_iter().map(|Pipeline(x, y)| (x, y))
    }
}

// 1. Break graph into pipelines
//   e.g.,  source -> filter -> shuffle-sink  ->  shuffle-source -> map -> sink
// becomes (source -> filter -> shuffle-sink) -> (shuffle-source -> map -> sink)
// NOTE: Currently we assume the graph is linear,
pub(crate) fn lower(mut graph: Graph2) -> Graph3 {
    let mut pipelines = HashMap::new();
    for (node_id, NodeId(pipeline_id)) in datalog::run(&graph) {
        pipelines
            .entry(PipelineId(pipeline_id))
            .or_insert_with(Pipeline2::default)
            .nodes
            .insert(node_id, graph.nodes.remove(&node_id).unwrap());
    }
    Graph3 {
        code: graph.code,
        pipelines,
    }
}

impl Node2 {
    pub fn data_inputs(&self) -> impl Iterator<Item = &NodeId> + '_ {
        match self {
            Node2::Filter { input, .. } => [Some(input), None],
            Node2::Map { input, .. } => [Some(input), None],
            Node2::KafkaSource { .. } => [None, None],
            Node2::KafkaSink { input, .. } => [Some(input), None],
            Node2::ShuffleSink { input, .. } => [Some(input), None],
            Node2::ShuffleSource { .. } => [None, None],
            Node2::Union { input0, input1, .. } => [Some(input0), Some(input1)],
            Node2::Window { input } => [Some(input), None],
        }
        .into_iter()
        .flatten()
    }

    pub fn is_sink(&self) -> bool {
        matches!(self, Node2::KafkaSink { .. } | Node2::ShuffleSink { .. })
    }

    pub fn is_source(&self) -> bool {
        matches!(
            self,
            Node2::KafkaSource { .. } | Node2::ShuffleSource { .. }
        )
    }
}
