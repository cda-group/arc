use crate::ast::Ident;
use petgraph::dot::{Config, Dot};
use petgraph::prelude::{Directed, Graph, NodeIndex};
use std::io::Write;
use std::process::{Command, Stdio};

#[derive(Debug)]
pub struct Node {
    id: Ident,
}

#[derive(Debug)]
pub struct Edge {
    source_port: usize,
    sink_port: usize,
}

#[derive(Default)]
pub struct Dataflow {
    graph: Graph<Node, Edge, Directed>,
}

impl Dataflow {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_node(&mut self, node: Node) -> NodeIndex {
        self.graph.add_node(node)
    }
    pub fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, edge: Edge) {
        self.graph.add_edge(a, b, edge);
    }
    pub fn pretty(&self) -> String {
        let config = &[Config::EdgeNoLabel];
        let dot = Dot::with_config(&self.graph, config);
        let mut child = Command::new("graph-easy")
            .arg("--as_ascii")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("graph-easy: Execution failed");
        child
            .stdin
            .as_mut()
            .expect("graph-easy: Failed to open stdin")
            .write_all(format!("{:?}", dot).as_bytes())
            .expect("graph-easy: Failed to write to stdin");
        let output = child
            .wait_with_output()
            .expect("graph-easy: Failed to read stdout");
        String::from_utf8_lossy(&output.stdout).to_string()
    }
}
