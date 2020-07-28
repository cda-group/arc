use smol_str::SmolStr;
use DimKind::*;
use ExprKind::*;
use LitKind::*;
use ShapeKind::*;
use TypeKind::*;
use {codespan::Span, derive_more::Constructor};
use crate::info::Info;

pub type ByteIndex = usize;

pub struct Spanned<T>(pub ByteIndex, pub T, pub ByteIndex);

pub type Symbol<'i> = &'i str;
pub type Name = SmolStr;
pub type Clause = (Pattern, Expr);
pub type Field = Name;

#[derive(Constructor)]
pub struct Script<'i> {
    pub tydefs: Vec<TypeDef>,
    pub fundefs: Vec<FunDef>,
    pub body: Expr,
    pub info: Info<'i>,
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
    pub ty: Type,
}

pub struct Decl {
    pub name: Name,
    pub ty: Type,
    pub kind: DeclKind,
}

impl Decl {
    pub fn new(symbol: Symbol, ty: Type, kind: DeclKind) -> Self {
        let name = symbol.into();
        Self { name, ty, kind }
    }
}

pub enum DeclKind {
    FunDecl,
    VarDecl,
    TypeDecl,
}

#[derive(Debug, Constructor, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub ty: Type,
    pub span: Span,
}

impl From<Spanned<ExprKind>> for Expr {
    fn from(Spanned(l, kind, r): Spanned<ExprKind>) -> Expr {
        let span = Span::new(l as u32, r as u32);
        let ty = Type {
            var: TypeVar(0),
            kind: Unknown,
            span: None,
        };
        Expr { kind, ty, span }
    }
}

impl Default for Expr {
    fn default() -> Expr { Expr::new(ExprErr, Type::new(), Span::new(0, 0)) }
}

#[derive(Debug, Eq, PartialEq, Clone, Copy, Hash)]
pub struct Ident(pub usize);

#[derive(Debug, Eq, PartialEq, Clone, Hash)]
pub struct Index(pub usize);

#[derive(Debug, Clone)]
pub enum ExprKind {
    Lit(LitKind),
    ConsArray(Vec<Expr>),
    ConsStruct(Vec<(Ident, Expr)>),
    ConsTuple(Vec<Expr>),
    Var(Ident),
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
    BinOpErr,
}

#[derive(Debug, Clone)]
pub enum BIFKind {
    Dataset(Box<Expr>),
    Fold(Box<Expr>, Box<Expr>),
    Fmap(Box<Expr>),
    Imap(Type, Box<Expr>),
}

#[derive(Debug, Clone)]
pub enum UnOpKind {
    Not,
    Cast(Type),
    MethodCall(Ident, Vec<Expr>),
    Access(Field),
    Project(Index),
    UnOpErr,
}

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub struct TypeVar(pub u32);

#[derive(Debug, Eq, Clone)]
pub struct Type {
    pub kind: TypeKind,
    pub var: TypeVar,
    pub span: Option<Span>,
}

impl PartialEq for Type {
    fn eq(&self, other: &Self) -> bool { self.kind == other.kind }
}

impl From<Spanned<TypeKind>> for Type {
    fn from(Spanned(l, kind, r): Spanned<TypeKind>) -> Type {
        let span = Some(Span::new(l as u32, r as u32));
        let var = TypeVar(0);
        Type { kind, var, span }
    }
}

impl Type {
    pub fn new() -> Type {
        let kind = Unknown;
        let span = None;
        let var = TypeVar(0);
        Type { kind, var, span }
    }

    pub fn fun(arity: usize) -> Type {
        let kind = Fun(
            (0..arity).map(|_| Type::new()).collect(),
            Box::new(Type::new()),
        );
        let span = None;
        let var = TypeVar(0);
        Type { kind, var, span }
    }
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum TypeKind {
    Scalar(ScalarKind),
    Optional(Box<Type>),
    Struct(Vec<(Ident, Type)>),
    Array(Box<Type>, Shape),
    Tuple(Vec<Type>),
    Fun(Vec<Type>, Box<Type>),
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
}

#[derive(Debug, Clone)]
pub struct Shape {
    pub kind: ShapeKind,
    pub span: Option<Span>,
}

impl Shape {
    pub fn simple(size: i32, span: Span) -> Shape {
        Shape {
            kind: Ranked(vec![Dim::known(size, span)]),
            span: None,
        }
    }

    pub fn unranked() -> Shape {
        Shape {
            kind: Unranked,
            span: None,
        }
    }
}

impl From<Spanned<ShapeKind>> for Shape {
    fn from(Spanned(l, kind, r): Spanned<ShapeKind>) -> Shape {
        let span = Some(Span::new(l as u32, r as u32));
        Shape { kind, span }
    }
}

impl PartialEq for Shape {
    fn eq(&self, other: &Self) -> bool {
        match (&self.kind, &other.kind) {
            (Unranked, _) | (_, Unranked) => true,
            (Ranked(d1), Ranked(d2)) => d1.len() == d2.len(),
        }
    }
}

impl Eq for Shape {}

#[derive(Debug, Clone)]
pub enum ShapeKind {
    Unranked,
    Ranked(Vec<Dim>),
}

#[derive(Debug, Clone)]
pub struct Dim {
    pub kind: DimKind,
    pub span: Option<Span>,
}

impl Dim {
    pub fn known(size: i32, span: Span) -> Dim {
        let kind = Symbolic(Expr::new(Lit(LitI32(size)), Type::new(), span));
        let span = Some(span);
        Dim { kind, span }
    }

    pub fn new() -> Dim {
        let kind = Hole;
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

#[derive(Debug, Clone)]
pub enum DimKind {
    Symbolic(Expr),
    Hole,
}

#[derive(Debug, Copy, Clone, Educe)]
#[educe(Hash)]
pub enum LitKind {
    LitI32(i32),
    LitI64(i64),
    LitF32(#[educe(Hash(ignore))] f32),
    LitF64(#[educe(Hash(ignore))] f64),
    LitBool(bool),
    LitErr,
}

impl PartialEq for LitKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (LitI32(v1), LitI32(v2)) => v1 == v2,
            (LitI64(v1), LitI64(v2)) => v1 == v2,
            (LitBool(v1), LitBool(v2)) => v1 == v2,
            _ => false,
        }
    }
}
impl Eq for LitKind {}
