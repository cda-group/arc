use diagnostics::Diagnostics;
use hir::*;
use im_rc::vector;
use im_rc::Vector;
use info::Info;
use name_gen::NameGen;
use stack::Stack;

#[derive(Clone, Debug)]
pub struct Context {
    pub(crate) stack: Stack<ScopeKind, ExprDecl, TypeDecl>,
    pub(crate) names: NameGen,
    pub(crate) stmts: Vector<Stmt>,
    pub diagnostics: Diagnostics,
}

#[derive(Clone, Debug)]
pub(crate) enum ScopeKind {
    Def(Vector<Generic>),
    Fun,
    Other,
}

#[derive(Clone, Debug)]
pub(crate) enum ExprDecl {
    Def(Vector<Generic>),
    Val(Info, Type),
}

#[derive(Clone, Debug)]
pub(crate) enum TypeDecl {}

impl Default for Context {
    fn default() -> Self {
        Self {
            stack: Stack::new(ScopeKind::Other),
            names: NameGen::new("_f"),
            stmts: vector![],
            diagnostics: Diagnostics::default(),
        }
    }
}

impl Context {
    pub fn new() -> Self {
        Self::default()
    }

    // Generic is captured by this function
    // Get all generics in scope

    pub(crate) fn generics_in_scope(&self) -> Vector<Generic> {
        self.stack
            .iter()
            .filter_map(|s| match &s.kind {
                ScopeKind::Def(gs) => Some(gs.clone()),
                ScopeKind::Fun => None,
                ScopeKind::Other => None,
            })
            .flatten()
            .collect()
    }

    pub(crate) fn values_in_scope(&self) -> Vector<ExprDecl> {
        todo!()
        // self.stack
        //     .iter()
        //     .filter_map(|s| {
        //         s.expr_namespace.iter().filter_map(|(x, k)| match k {
        //             ExprDecl::Def(_) => None,
        //             ExprDecl::Val(_, t) => Some(x),
        //         })
        //     })
        //     .collect()
    }
}
