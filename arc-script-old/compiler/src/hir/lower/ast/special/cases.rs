use super::Context;
use crate::ast;
use crate::hir;
use crate::hir::Name;
use arc_script_compiler_shared::Shrinkwrap;

use pattern_compiler::CfgNodeKind;
use pattern_compiler::EdgeRef;
use pattern_compiler::ExpandedClauseNodes;
use pattern_compiler::PatternCfg as PatCfg;
use pattern_compiler::PatternProvider as PatProvider;
use petgraph_4::graph::NodeIndex as PatternNodeKey;

use petgraph_4::Direction;
use petgraph_4::Graph;

/// This module is compiles patterns into more basic expressions while also performing
/// exhaustiveness checks. Patterns are a form of syntactic sugar which can occur
/// wherever variables are bound. Variables can be bound in the following places:
///
/// * let <pat> = <expr> in <expr>
/// * match <expr> { (<pat> => <expr>)* }
/// * |<pat>*| <expr>
/// * fun <id>(<pat>*) <expr> end
/// * task <id>(<pat>*) <item>* on (<pat> => <expr>)* end
///
/// The above patterns can further be classified as refutable and irrefutable:
///
/// * Irrefutable: let, closure, fun, task
/// * Refutable: match
///
/// For irrefutable patterns, the idea is to compile the pattern into a chain of SSA instructions:
///
/// ```txt
///     let ((a,b),c) = d in
///     e
/// ```
///
/// becomes:
///
/// ```txt
///     let x0 =  d.0 in
///     let a  = x0.0 in
///     let b  = x0.1 in
///     let c  =  d.1 in
///     e
/// ```
///
/// For refutable patterns, the idea is to compile the pattern into a decision tree:
///
/// ```txt
///     match e0 {
///         (1,_) => e1,
///         (_,2) => e2,
///         (_,_) => e3,
///     }
/// ```
///
/// becomes:
///
/// ```txt
///     let x0 = e0.0 in
///     if x0 == 1 then
///         e1
///     else
///         if x1 == 2 then
///             e2
///         else
///             e3
/// ```
///
/// Refutable patterns may also contain guards:
///
/// ```txt
///     match e0 {
///         (1,_) if foo() => e1,
///         (_,2) if bar() => e2,
///         (_,_) if baz() => e3,
///     }
/// ```
///
/// becomes
///
/// ```txt
///     let x0 = e0.0 in
///     if x0 == 1 then
///         if foo() then
///             e1
///         else
///             let x1 = e0.1 in
///             if x2 == 1 then
///                 if bar() then
///                     e2
///                 else
///                     e3
///     else
///         let x1 = e0.1 in
///         if x2 == 1 then
///             if bar() then
///                 e2
///             else
///                 e3
///         else
///             e3
/// ```
pub(crate) fn lower_cases(cases: &[ast::Case], _ctx: &mut Context<'_>) -> hir::ExprKind {
    let mut pattern_tree = PatternTree::default();
    for case in cases {
        let clause = pattern_tree.construct_tree(&case.pat);
        pattern_tree.clauses.push(clause);
    }
    println!("{:#?}", pattern_tree);
    let decision_tree = pattern_compiler::to_decision_tree(&mut pattern_tree);
    let _root = decision_tree.entry;
    println!("{:#?}", decision_tree);
    //     let expr = convert_to_expr(decision_tree.entry);
    for node_index in decision_tree.graph.node_indices() {
        let node = &decision_tree.graph[node_index];
        match node {
            CfgNodeKind::Root => {}
            CfgNodeKind::Match(_x) => {}
            CfgNodeKind::Fail => {}
            CfgNodeKind::Leaf(_x) => {}
        }
        if let Some(_bindings) = decision_tree.leaf_bindings.get(&node_index) {}
    }
    crate::todo!()
}

// fn recurse(node: CfgNodeKey, cfg: &pattern_compiler::PatternCfg<PatternTree>) {
//     for edge in cfg.graph
// }

impl PatternTree {
    /// Converts an AST-pattern into a decision tree.
    fn construct_tree(&mut self, pat: &ast::Pat) -> PatternNodeKey {
        match &pat.kind {
            ast::PatKind::Tuple(ps) => {
                let node = self.pattern.add_node(PatternNodeKind::Tuple(ps.len()));
                for p in ps.iter() {
                    let child = self.construct_tree(p);
                    self.pattern.add_edge(node, child, ());
                }
                node
            }
            ast::PatKind::Var(x) => self.pattern.add_node(PatternNodeKind::Var(*x)),
            ast::PatKind::Ignore => self.pattern.add_node(PatternNodeKind::Ignore),
            ast::PatKind::Val(_v) => self.pattern.add_node(PatternNodeKind::Val),
            ast::PatKind::Or(p1, p2) => {
                let node = self.pattern.add_node(PatternNodeKind::Or);
                let child1 = self.construct_tree(p1);
                let child2 = self.construct_tree(p2);
                self.pattern.add_edge(node, child1, ());
                self.pattern.add_edge(node, child2, ());
                node
            }
            ast::PatKind::Err => self.pattern.add_node(PatternNodeKind::Err),
            ast::PatKind::Struct(_) => crate::todo!(),
            ast::PatKind::Variant(_, _) => crate::todo!(),
        }
    }

    /// The output CFG contains four kinds of nodes:
    ///
    /// - One single Root node. This is where the decision tree starts.
    ///   The id of this node is stored in the `entry` field. This has a
    ///   single outgoing edge, usually pointing at a Match node.
    ///
    /// - Any number of Match nodes. These nodes also take a Variable
    ///   as an argument, any outgoing edges from this is matched on that
    ///   variable. Outgoing edges from a Match node always contain:
    ///   - A `kind`, the kind that the value must match for this branch
    ///     to be taken.
    ///   - `variable_binds`, the expansions of that matched value. An
    ///     example would be a `cons`, which would expand to two values,
    ///     `x`, the head and `y`, the tail.
    ///
    /// - M number of Leaf nodes, where M is the number of clauses in the
    ///   input pattern. Ending up at this node is a successful match on
    ///   that clause.
    ///
    /// - Fail nodes, ending up in this node means no clauses were matched.
    ///
    /// `leaf_bindings` contains a map for every node in the CFG, which maps
    /// each visible `PatternNodeKey` of the input pattern to the `CfgVariable`
    /// that represents it. This enables you to refer back to things that
    /// were matched on in the pattern when lowering the resulting CFG back
    /// into your compiler IR.
    ///
    /// Exhaustiveness means the pattern is irrefutable.
    /// A pattern is refutable if the `DecisionTree` contains a
    /// * Match-node which has a child Fail-node
    /// * Leaf-node which has a child Fail-Node which is guarded
    fn is_exhaustive(tree: &DecisionTree) -> bool {
        for edge in tree.graph.edge_references() {
            let parent = &tree.graph[edge.source()];
            let child = &tree.graph[edge.target()];
            if let (CfgNodeKind::Match(_), CfgNodeKind::Fail) = (parent, child) {
                return false;
            }
        }
        true
    }
}

type DecisionTree = PatCfg<PatternTree>;
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, Shrinkwrap)]
#[shrinkwrap(mutable)]
pub(crate) struct CfgVar(pub(crate) usize);

impl CfgVar {
    const fn new(v: usize) -> Self {
        Self(v)
    }
}

#[derive(Copy, Clone, Hash, Debug, PartialEq, Eq)]
pub(crate) enum PatternNodeKind {
    Tuple(usize), // (a, b, c)
    Var(Name),    // a
    Val,          // 1
    Or,           // a | b
    Ignore,       // _
    Struct,       // { a, b, c }
    Variant,      // A(a)
    Err,
}

pub(crate) const ROOT: CfgVar = CfgVar::new(0);

#[derive(Debug)]
pub(crate) struct PatternTree {
    pattern: Graph<PatternNodeKind, ()>,
    /// Top-level clauses of the pattern
    clauses: Vec<PatternNodeKey>,
    counter: CfgVar,
    ignore: PatternNodeKey,
    bodies: Vec<ast::Expr>,
}

impl PatternTree {
    fn next_var(&mut self) -> CfgVar {
        *self.counter += 1;
        self.counter
    }
}

impl Default for PatternTree {
    fn default() -> Self {
        let mut pattern = Graph::new();
        Self {
            ignore: pattern.add_node(PatternNodeKind::Ignore),
            pattern,
            clauses: Vec::new(),
            counter: CfgVar::new(0),
            bodies: Vec::new(),
        }
    }
}

impl PatProvider for PatternTree {
    type PatternNodeKey = PatternNodeKey;
    type PatternNodeKind = PatternNodeKind;
    type CfgVariable = CfgVar;

    const WILDCARD: PatternNodeKind = PatternNodeKind::Ignore;

    fn get_root(&self) -> ExpandedClauseNodes<CfgVar, PatternNodeKey> {
        ExpandedClauseNodes {
            variables: vec![ROOT],
            clauses: self.clauses.len(),
            nodes: self.clauses.clone(),
        }
    }

    fn kind_includes(&self, kind: PatternNodeKind, key: PatternNodeKey) -> bool {
        self.pattern[key] == kind
    }

    fn get_kind(&self, key: PatternNodeKey) -> PatternNodeKind {
        self.pattern[key]
    }

    fn get_wildcard_node(&self) -> PatternNodeKey {
        self.ignore
    }

    fn expand_clause_nodes(
        &mut self,
        nodes: Vec<PatternNodeKey>,
        kind: PatternNodeKind,
    ) -> ExpandedClauseNodes<CfgVar, PatternNodeKey> {
        let mut expanded = ExpandedClauseNodes {
            variables: vec![],
            clauses: nodes.len(),
            nodes: vec![],
        };

        // Base case, no more clauses to expand
        if nodes.is_empty() {
            return expanded;
        }

        // Expand
        let count = self
            .pattern
            .edges_directed(nodes[0], Direction::Outgoing)
            .count();

        for _ in 0..count {
            expanded.variables.push(self.next_var());
        }

        match kind {
            PatternNodeKind::Tuple(_) => {
                for node in &nodes {
                    for edge in self.pattern.edges_directed(*node, Direction::Outgoing) {
                        let child = edge.target();
                        expanded.nodes.push(child);
                    }
                }
            }
            PatternNodeKind::Ignore | PatternNodeKind::Var(_) | PatternNodeKind::Val => {}
            x => unimplemented!("{:?}", x),
        }

        expanded
    }
}
