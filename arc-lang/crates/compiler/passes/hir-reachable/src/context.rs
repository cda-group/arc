use diagnostics::Diagnostics;
use hir::*;
use im_rc::vector;
use im_rc::HashSet;
use im_rc::OrdMap;
use im_rc::Vector;
use info::Info;
use stack::Stack;
use std::rc::Rc;

#[derive(Debug)]
pub struct Context {
    pub(crate) stack: Stack<(), ExprDecl, TypeDecl>,
    pub diagnostics: Diagnostics,
    pub(crate) reachable: HashSet<Name>,
    pub(crate) stmts: Vector<Stmt>,
}

#[derive(Clone, Debug)]
pub(crate) enum ExprDecl {
    Def(Info, Meta, Vector<Pattern>, Type, Block),
    Bif(Info, Meta, Vector<Type>, Type),
}

#[derive(Clone, Debug)]
pub(crate) enum TypeDecl {
    Enum(Info, Meta, Vector<(Name, Type)>),
    Bit(Info, Meta),
}

impl Default for Context {
    fn default() -> Self {
        Self {
            stack: Stack::new(()),
            diagnostics: Diagnostics::default(),
            reachable: HashSet::new(),
            stmts: Vector::new(),
        }
    }
}

impl Context {
    pub fn new() -> Self {
        Self::default()
    }
}
