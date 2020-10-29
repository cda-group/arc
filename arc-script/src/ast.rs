use crate::{info::Info, typer::Typer};
use chrono::Duration;
use lasso::Spur;
use DimKind::*;
use ExprKind::*;
use LitKind::*;
use TypeKind::*;
use {codespan::Span, derive_more::Constructor};

pub type ByteIndex = usize;

pub struct Spanned<T>(pub ByteIndex, pub T, pub ByteIndex);

pub type SymbolName<'i> = &'i str;
pub type SymbolKey = Spur;
pub type Clause = (Pattern, Expr);
pub type Field = SymbolKey;

#[derive(Constructor)]
pub struct Script<'i> {
    pub ast: SyntaxTree,
    pub info: Info<'i>,
}

#[derive(Constructor)]
pub struct SyntaxTree {
    pub taskdefs: Vec<TaskDef>,
    pub tydefs: Vec<TypeDef>,
    pub fundefs: Vec<FunDef>,
    pub body: Expr,
}

#[derive(Constructor)]
pub struct FunDef {
    pub id: Ident,
    pub params: Vec<Ident>,
    pub body: Expr,
}

#[derive(Constructor)]
pub struct TypeDef {
    pub id: Ident,
    pub tv: TypeVar,
}

/// A task is a generic low-level primitive which resembles a node in the dataflow graph.
///
/// Tasks may be specialized into any higher order function (e.g., map, filter) by implementing
/// three callback functions:
/// * setup():             invoked when the operator is created (optional).
/// * process(self, elem): invoked on each arriving data stream element (required).
/// * trigger(self):       invoked when a timer is triggered (optional).
///
/// In addition, the operator itself has builtin methods for controlling its functionality:
/// * self.send(<channel>, <data>):   sends <data> over <channel>
/// * self.schedule(<time>):          schedules a timer to trigger at a given <time>
/// * self.update(<field> = <state>): updates <field> to contain a new <state>
///
/// ```text
/// task map(input, fun) -> (output)
///     fun process(self, elem) = self.output(self.fun(elem))
/// end
///
/// task filter(input, fun) -> (output)
///     fun process(self, elem) =
///         if fun(elem)
///         then self.send(output, elem)
///         else self
/// end
///
/// task flat_map(input, fun) -> (output)
///     fun process(self, elem) =
///         fun(elem).fold(self, (self, elem) => self.send(output, elem))
/// end
///
/// task partition(input, fun) -> (output1, output2)
///     fun process(self, elem) =
///         if fun(elem)
///         then self.send(output1, elem)
///         else self.send(output2, elem)
/// end
/// ```
#[derive(Constructor)]
pub struct TaskDef {
    pub id: Ident,
    pub params: Vec<Ident>,
    pub fundefs: Vec<FunDef>,
}

#[derive(Constructor)]
pub struct Decl {
    pub sym: SymbolKey,
    pub tv: TypeVar,
    pub kind: DeclKind,
}

pub enum DeclKind {
    FunDecl,
    VarDecl,
    TypeDecl,
    TaskDecl(Vec<Ident>),
}

#[derive(Debug, Clone, Constructor)]
pub struct Expr {
    pub kind: ExprKind,
    pub tv: TypeVar,
    pub span: Span,
}

impl Expr {
    pub fn from(Spanned(l, kind, r): Spanned<ExprKind>, typer: &mut Typer) -> Expr {
        let span = Span::new(l as u32, r as u32);
        let tv = typer.fresh();
        Expr { kind, tv, span }
    }
    /// Moves an expression out of its mutable reference, replacing it with an error expression
    pub fn take(&mut self) -> Expr {
        std::mem::replace(self, Expr::new(ExprErr, TypeVar(0), Span::new(0, 0)))
    }
}

#[derive(Debug, Eq, PartialEq, Clone, Copy, Hash)]
pub struct Ident(pub usize);

#[derive(Debug, Eq, PartialEq, Clone, Hash)]
pub struct Index(pub usize);

#[derive(Debug, Clone)]
pub enum ExprKind {
    Lit(LitKind),
    ConsArray(Vec<Expr>),
    ConsStruct(Vec<(SymbolKey, Expr)>),
    ConsTuple(Vec<Expr>),
    Var(Ident),
    Closure(Vec<Ident>, Box<Expr>),
    UnOp(UnOpKind, Box<Expr>),
    BinOp(Box<Expr>, BinOpKind, Box<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    Let(Ident, Box<Expr>, Box<Expr>),
    Match(Box<Expr>, Vec<Clause>),
    FunCall(Ident, Vec<Expr>),
    ExprErr,
}

#[derive(Debug, Clone)]
pub struct Pattern {
    pub vars: Vec<Ident>,
    pub kind: PatternKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum PatternKind {
    Regex(regex::Regex),
    DeconsTuple(Vec<Pattern>),
    Val(LitKind),
    Bind(Ident),
    Or(Box<Pattern>, Box<Pattern>),
    Wildcard,
    PatternErr,
}

impl From<Spanned<PatternKind>> for Pattern {
    fn from(Spanned(l, kind, r): Spanned<PatternKind>) -> Pattern {
        let span = Span::new(l as u32, r as u32);
        let vars = vec![]; // TODO: Extract variables from patterns
        Pattern { kind, vars, span }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Neq,
    Gt,
    Lt,
    Geq,
    Leq,
    BinOpErr,
}

#[derive(Debug, Clone)]
pub enum BIFKind {
    Dataset(Box<Expr>),
    Fold(Box<Expr>, Box<Expr>),
    Fmap(Box<Expr>),
    Imap(TypeVar, Box<Expr>),
}

#[derive(Debug, Clone)]
pub enum UnOpKind {
    Not,
    Cast(TypeVar),
    MethodCall(Ident, Vec<Expr>),
    Access(Field),
    Project(Index),
    UnOpErr,
}

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub struct TypeVar(pub u32);

#[derive(Debug, Eq, Clone, Educe)]
#[educe(PartialEq)]
pub struct Type {
    pub kind: TypeKind,
    #[educe(PartialEq(ignore))]
    pub span: Option<Span>,
}

impl From<Spanned<TypeKind>> for Type {
    fn from(Spanned(l, kind, r): Spanned<TypeKind>) -> Type {
        let span = Some(Span::new(l as u32, r as u32));
        Type { kind, span }
    }
}

impl From<TypeKind> for Type {
    fn from(kind: TypeKind) -> Type {
        let span = None;
        Type { kind, span }
    }
}

impl Type {
    pub fn new() -> Type {
        let kind = Unknown;
        let span = None;
        Type { kind, span }
    }

    pub fn fun(arity: usize, typer: &mut Typer) -> Type {
        let input = (0..arity).map(|_| typer.fresh()).collect();
        let output = typer.fresh();
        let kind = Fun(input, output);
        let span = None;
        Type { kind, span }
    }
}

#[derive(Debug, Eq, Clone, PartialEq)]
pub enum TypeKind {
    Scalar(ScalarKind),
    Optional(TypeVar),
    Struct(Vec<(SymbolKey, TypeVar)>),
    Array(TypeVar, Shape),
    Tuple(Vec<TypeVar>),
    Fun(Vec<TypeVar>, TypeVar),
    Stream(TypeVar),
    Unknown,
    TypeErr,
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum ScalarKind {
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    Bool,
    Null,
    Str,
    Unit,
}

#[derive(Debug, Eq, Clone, Educe)]
#[educe(PartialEq)]
pub struct Shape {
    pub dims: Vec<Dim>,
    #[educe(PartialEq(ignore))]
    pub span: Option<Span>,
}

impl From<Spanned<Vec<Dim>>> for Shape {
    fn from(Spanned(l, dims, r): Spanned<Vec<Dim>>) -> Shape {
        let span = Some(Span::new(l as u32, r as u32));
        Shape { dims, span }
    }
}

impl From<Vec<Dim>> for Shape {
    fn from(dims: Vec<Dim>) -> Shape {
        let span = None;
        Shape { dims, span }
    }
}

#[derive(Debug, Eq, Clone, Educe)]
#[educe(PartialEq)]
pub struct Dim {
    pub kind: DimKind,
    #[educe(PartialEq(ignore))]
    pub span: Option<Span>,
}

impl From<DimKind> for Dim {
    fn from(kind: DimKind) -> Dim {
        let span = None;
        Dim { kind, span }
    }
}

impl Dim {
    pub fn is_val(&self) -> bool {
        if let DimVal(_) = self.kind {
            true
        } else {
            false
        }
    }
}

/// An expression solveable by the z3 SMT solver
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimKind {
    DimVar(i32),
    DimVal(i32),
    DimOp(Box<Dim>, DimOpKind, Box<Dim>),
    DimErr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimOpKind {
    DimAdd,
    DimSub,
    DimMul,
    DimDiv,
}

impl Dim {
    pub fn known(size: i32, span: Span) -> Dim {
        let kind = DimVal(size);
        let span = Some(span);
        Dim { kind, span }
    }

    pub fn new() -> Dim {
        let kind = DimVar(0);
        let span = None;
        Dim { kind, span }
    }
}

impl From<Spanned<DimKind>> for Dim {
    fn from(Spanned(l, kind, r): Spanned<DimKind>) -> Dim {
        let span = Some(Span::new(l as u32, r as u32));
        Dim { kind, span }
    }
}

#[derive(Debug, Copy, Clone, Educe)]
#[educe(Hash)]
pub enum LitKind {
    LitI8(i8),
    LitI16(i16),
    LitI32(i32),
    LitI64(i64),
    LitF32(#[educe(Hash(ignore))] f32),
    LitF64(#[educe(Hash(ignore))] f64),
    LitBool(bool),
    LitTime(Duration),
    LitErr,
}

impl PartialEq for LitKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (LitI8(v1), LitI8(v2)) => v1 == v2,
            (LitI16(v1), LitI16(v2)) => v1 == v2,
            (LitI32(v1), LitI32(v2)) => v1 == v2,
            (LitI64(v1), LitI64(v2)) => v1 == v2,
            (LitBool(v1), LitBool(v2)) => v1 == v2,
            _ => false,
        }
    }
}

impl Eq for LitKind {}
