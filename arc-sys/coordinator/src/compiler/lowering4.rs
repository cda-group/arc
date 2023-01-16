use std::collections::HashSet;

use halfbrown::HashMap;

use super::lowering2::Node2;
use super::lowering2::NodeId;
use super::lowering3::Graph3;
use super::lowering3::Pipeline2;
use super::lowering3::PipelineId;

#[derive(Clone, Debug)]
pub struct Graph4 {
    pub code: String,
    pub pipelines: HashMap<PipelineId, Pipeline3>,
}

#[derive(Clone, Debug)]
pub struct Pipeline3 {
    pub nodes: Vec<(NodeId, Node2)>,
}

pub fn lower(graph: Graph3) -> Graph4 {
    let pipelines = graph
        .pipelines
        .into_iter()
        .map(|(id, pipeline)| (id, topological_sort(pipeline)))
        .collect();
    Graph4 {
        code: graph.code,
        pipelines,
    }
}

// Topologically sort the nodes in the pipeline
pub fn topological_sort(mut pipeline: Pipeline2) -> Pipeline3 {
    let mut nodes = Vec::new();
    let mut visited = HashSet::new();

    let mut stack = pipeline
        .nodes
        .iter()
        .filter(|(_, node)| node.is_sink())
        .map(|(id, _)| id)
        .copied()
        .collect::<Vec<_>>();

    while let Some(id) = stack.pop() {
        visited.insert(id);
        nodes.push(id);
        match pipeline.nodes.get(&id).unwrap() {
            Node2::Filter { input, .. } => {
                if !visited.contains(input) {
                    stack.push(*input);
                }
            }
            Node2::Map { input, .. } => {
                if !visited.contains(input) {
                    stack.push(*input);
                }
            }
            Node2::KafkaSource { .. } => {}
            Node2::KafkaSink { input, .. } => {
                if !visited.contains(input) {
                    stack.push(*input);
                }
            }
            Node2::ShuffleSink { input, .. } => {
                if !visited.contains(input) {
                    stack.push(*input);
                }
            }
            Node2::ShuffleSource { .. } => {}
            Node2::Union { input0, input1 } => {
                if !visited.contains(input0) {
                    stack.push(*input0);
                }
                if !visited.contains(input0) {
                    stack.push(*input1);
                }
            }
            Node2::Window { input } => {
                if !visited.contains(input) {
                    stack.push(*input);
                }
            }
        }
    }
    nodes.reverse();
    Pipeline3 {
        nodes: nodes
            .into_iter()
            .map(|id| (id, pipeline.nodes.remove(&id).unwrap()))
            .collect(),
    }
}
