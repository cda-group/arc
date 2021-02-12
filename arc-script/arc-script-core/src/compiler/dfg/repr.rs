use crate::compiler::dfg::from::eval::stack::Frame;
use crate::compiler::hir::Path;
use crate::compiler::hir::HIR;
use crate::compiler::info::names::NameId;
use crate::compiler::info::paths::PathId;
use crate::compiler::shared::New;

use petgraph::dot::{Config, Dot};
use petgraph::prelude::{Directed, Graph};
use shrinkwraprs::Shrinkwrap;

use std::io::Write;
use std::process::{Command, Stdio};
use std::rc::Rc;

pub(crate) use petgraph::prelude::EdgeIndex as EdgeId;
pub(crate) use petgraph::prelude::NodeIndex as NodeId;

#[derive(Debug, Copy, Clone, New)]
pub struct Node {
    pub(crate) id: NodeId,
}

#[derive(Debug, New)]
pub(crate) struct NodeData {
    /// Path to the task corresponding to this node.
    pub(crate) path: Path,
    pub(crate) frame: Frame,
}

pub(crate) type Port = usize;

#[derive(New, Debug)]
pub(crate) struct EdgeData {
    /// Port which this edge connects from.
    pub(crate) oport: Port,
    /// Port which this edge connects to.
    pub(crate) iport: Port,
}

#[derive(New, Debug, Default, Shrinkwrap)]
#[shrinkwrap(mutable)]
pub(crate) struct DFG {
    pub(crate) graph: Graph<NodeData, EdgeData, Directed>,
}

impl DFG {
    /// Displays the dataflow graph in ascii format.
    pub(crate) fn ascii(&self) -> String {
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
