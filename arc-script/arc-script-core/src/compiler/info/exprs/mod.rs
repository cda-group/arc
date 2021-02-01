use crate::repr::ast::Spanned;
use crate::repr::hir::VecMap;
use crate::repr::info::files::Loc;
use crate::repr::info::names::NameId;
use crate::repr::info::paths::PathId;
use crate::repr::info::types::TypeId;
use spanned_derive::Spanned;

/// A data structure which stores all expressions in the program.
/// Expressions are accessed through indexing.
#[derive(Debug, Default)]
pub(crate) struct ExprInterner {
    store: Vec<ExprInfo>,
}

impl ExprInterner {
    /// Interns an expression and returns its id.
    fn intern(&mut self, expr: ExprInfo) -> ExprId {
        self.store.push(expr);
        ExprId(self.store.len() - 1)
    }

    fn resolve(&self, id: ExprId) -> &ExprInfo {
        self.store.get(*id).unwrap()
    }

    fn resolve_mut(&mut self, id: ExprId) -> &ExprInfo {
        self.store.get_mut(*id).unwrap()
    }
}

#[derive(Debug)]
pub(crate) struct ExprInfo {
    tv: TypeId,
    kind: ExprKind,
}

#[derive(Debug, Shrinkwrap, Eq, PartialEq, Hash)]
pub(crate) struct ExprId(usize);

/// A kind of expression.
#[derive(Debug)]
pub(crate) enum ExprKind {
    Lit(Lit),
    Array(Vec<ExprId>),
    Struct(VecMap<NameId, ExprInfo>),
    Tuple(Vec<ExprId>),
    Path(PathId),
    UnOp(UnOp, ExprId),
    BinOp(ExprId, BinOp, ExprId),
    If(ExprId, ExprId, ExprId),
    Let(NameId, ExprId, ExprId),
    Emit(ExprId),
    Loop(ExprId, ExprId),
    Err,
}

#[derive(Debug)]
pub(crate) enum UnOp {
    Not,
    Neg,
    Access(NameId),
    Project(usize),
    Call(Vec<ExprId>),
    Err,
}

#[derive(Debug)]
pub(crate) enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Equ,
    Neq,
    Or,
    And,
    Gt,
    Lt,
    Geq,
    Leq,
    Pipe,
    Seq,
    Err,
}

#[derive(Debug)]
pub(crate) enum Lit {
    SInt(i128),
    UInt(u128),
    Float(f64),
    Bool(bool),
    Char(char),
    Unit,
    Err,
}
