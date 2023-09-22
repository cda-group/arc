use diagnostics::Diagnostics;
use hir::*;
use im_rc::OrdMap;
use im_rc::Vector;
use name_gen::NameGen;
use stack::Stack;

#[derive(Debug)]
pub struct Context {
    pub(crate) stack: Stack<Vector<Stmt>, Name, ()>,
    clauses: Vector<Clause>,
    pub(crate) vals: NameGen,
    pub diagnostics: Diagnostics,
}

impl Default for Context {
    fn default() -> Self {
        Context {
            stack: Stack::new(Vector::new()),
            clauses: Vector::new(),
            vals: NameGen::new("v"),
            diagnostics: Diagnostics::default(),
        }
    }
}

pub(crate) type Equations = OrdMap<Name, Pattern>;
pub(crate) type Substitutions = OrdMap<Name, Name>;

#[derive(Debug, Clone)]
pub(crate) struct Clause {
    pub(crate) eqs: Equations,
    pub(crate) substs: Substitutions,
    pub(crate) b: Block,
}

impl Clause {
    pub(crate) fn new(eqs: Equations, substs: Substitutions, b: Block) -> Self {
        Self { eqs, substs, b }
    }
}

impl Context {
    pub fn new() -> Context {
        Self::default()
    }

    pub(crate) fn add_expr(&mut self, e: Expr) -> Name {
        let info = e.info;
        let t = e.t.clone();
        let x = self.vals.fresh();
        let p = PVal(x.clone()).with(t.clone(), info);
        let s = StmtKind::SVal(p, e).with(info);
        self.add_stmt(s);
        x
    }

    pub(crate) fn add_expr_val(&mut self, e: Expr) -> Expr {
        let x = self.add_expr(e.clone());
        EVal(x).with(e.t, e.info)
    }

    pub(crate) fn add_stmt(&mut self, s: Stmt) {
        self.stack.current().push_back(s);
    }

    pub(crate) fn add_stmts(&mut self, ss: Vector<Stmt>) {
        self.stack.current().extend(ss);
    }
}
