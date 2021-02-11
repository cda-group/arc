//! Module for displaying the `DFG`.

#![allow(clippy::useless_format)]
use crate::compiler::dfg;
use crate::compiler::hir;
use crate::compiler::info::names::NameId;
use crate::compiler::info::paths::PathId;
use crate::compiler::info::types::TypeId;
use crate::compiler::info::Info;
use crate::compiler::shared::display::format::Context;
use crate::compiler::shared::display::pretty::*;
use crate::compiler::shared::New;

use petgraph::Direction;
use std::fmt::{self, Display, Formatter};

/// State needed to display the `DFG`.
#[derive(Copy, Clone, New)]
pub(crate) struct State<'i> {
    info: &'i Info,
}

/// Wraps the `DFG` inside a struct which can be pretty printed.
pub(crate) fn pretty<'i, 'j, Node>(node: &'i Node, info: &'j Info) -> Pretty<'i, Node, State<'j>> {
    node.to_pretty(State::new(info))
}

impl<'i> Display for Pretty<'i, dfg::DFG, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "-- DFG")?;
        let Pretty(dfg, ctx) = self;
        writeln!(f, "Sources:")?;
        for node in dfg.graph.externals(Direction::Incoming) {
            let node = dfg.graph.node_weight(node).unwrap();
            writeln!(f, "* {}", node.path.pretty(ctx));
        }
        writeln!(f, "Sinks:")?;
        for node in dfg.graph.externals(Direction::Outgoing) {
            let node = dfg.graph.node_weight(node).unwrap();
            writeln!(f, "* {}", node.path.pretty(ctx));
        }
        writeln!(f, "Edges:")?;
        for edge in dfg.graph.edge_indices() {
            let (origin, target) = dfg.graph.edge_endpoints(edge).unwrap();
            let edge = dfg.graph.edge_weight(edge).unwrap();
            let origin = dfg.graph.node_weight(origin).unwrap();
            let target = dfg.graph.node_weight(target).unwrap();
            writeln!(
                f,
                "* {}{{{}}} --> {{{}}}{}",
                origin.path.pretty(ctx),
                edge.oport.pretty(ctx),
                edge.iport.pretty(ctx),
                target.path.pretty(ctx)
            );
        }
        Ok(())
    }
}

impl<'i> Display for Pretty<'i, hir::Path, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(x, ctx) = self;
        todo!()
        //         write!(f, "{}", crate::compiler::hir::display::State::new(ctx.state).pretty(ctx))
    }
}

impl<'i> Display for Pretty<'i, dfg::Port, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(x, ctx) = self;
        todo!()
        //         write!(f, "{}", crate::compiler::hir::display::State::new(ctx.state).pretty(ctx))
    }
}
