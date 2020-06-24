use {crate::error::CompilerError, codespan::Span, derive_more::Constructor, smol_str::SmolStr};

type ByteIndex = usize;

pub struct Spanned<T>(pub ByteIndex, pub T, pub ByteIndex);

pub struct Script {
    pub funs: Vec<Fun>,
    pub body: Expr,
    pub errors: Vec<CompilerError>,
}

impl Script {
    pub fn new(funs: Vec<Fun>, body: Expr) -> Script {
        let errors = Vec::new();
        Script { funs, body, errors }
    }
}

#[derive(Constructor)]
pub struct Fun {
    pub id: Ident,
    pub params: Vec<(Ident, Type)>,
    pub body: Expr,
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
            kind: TypeKind::Unknown,
            span: None,
        };
        Expr { kind, ty, span }
    }
}

impl Default for Expr {
    fn default() -> Expr { Expr::new(ExprKind::Error, Type::new(), Span::new(0, 0)) }
}

pub type Scope = usize;

#[derive(Debug, Eq, PartialEq, Clone, Copy, Hash)]
pub struct Uid(pub u32);
impl Uid {
    pub fn reset() {
        unsafe {
            COUNTER = 0;
        }
    }

    pub fn new() -> Uid {
        unsafe {
            let uid = Uid(COUNTER);
            COUNTER += 1;
            uid
        }
    }
}

#[derive(Debug, Eq, PartialEq, Clone, Hash)]
pub struct Ident {
    pub name: SmolStr,
    pub scope: Option<Scope>,
    pub uid: Option<Uid>,
}

#[derive(Debug, Eq, PartialEq, Clone, Hash)]
pub struct Index(pub usize);

static mut COUNTER: u32 = 0;
impl Ident {
    pub fn from(name: impl Into<String> + AsRef<str>) -> Ident {
        Ident {
            name: SmolStr::new(name),
            scope: None,
            uid: None,
        }
    }
}

impl Ident {
    pub fn gen() -> Ident {
        Ident {
            name: SmolStr::new("x"),
            scope: None,
            uid: Some(Uid::new()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    Lit(Lit),
    Bif(Bif),
    Var(Ident),
    UnOp(UnOp, Box<Expr>),
    BinOp(Box<Expr>, BinOp, Box<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    Let(Ident, Type, Box<Expr>, Box<Expr>),
    Call(Ident, Vec<Expr>),
    Error,
}

#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Error,
}

#[derive(Debug, Clone)]
pub enum Bif {
    Dataset(Box<Expr>),
    Fold(Box<Expr>, Box<Expr>),
    Fmap(Box<Expr>),
    Imap(Type, Box<Expr>),
    Error,
}

#[derive(Debug, Clone)]
pub enum UnOp {
    Not,
    Cast(Type),
    Call(Ident, Vec<Expr>),
    Access(Ident),
    Project(Index),
    Error,
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
        let kind = TypeKind::Unknown;
        let span = None;
        let var = TypeVar(0);
        Type { kind, var, span }
    }
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum TypeKind {
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    Bool,
    Null,
    String,
    Option(Box<Type>),
    Struct(Vec<(Ident, Type)>),
    Array(Box<Type>, Shape),
    Unknown,
    Error,
}

#[derive(Debug, Clone)]
pub struct Shape {
    pub kind: ShapeKind,
    pub span: Option<Span>,
}

impl Shape {
    pub fn simple(size: i32, span: Span) -> Shape {
        Shape {
            kind: ShapeKind::Ranked(vec![Dim::known(size, span)]),
            span: None,
        }
    }

    pub fn unranked() -> Shape {
        Shape {
            kind: ShapeKind::Unranked,
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
            (ShapeKind::Unranked, _) | (_, ShapeKind::Unranked) => true,
            (ShapeKind::Ranked(d1), ShapeKind::Ranked(d2)) => d1.len() == d2.len(),
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
        let kind = DimKind::Expr(Expr::new(ExprKind::Lit(Lit::I32(size)), Type::new(), span));
        let span = Some(span);
        Dim { kind, span }
    }

    pub fn new() -> Dim {
        let kind = DimKind::Unknown;
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
    Expr(Expr),
    Unknown,
}

#[derive(Debug, Clone)]
pub enum Lit {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Array(Vec<Expr>),
    Struct(Vec<(Ident, Expr)>),
    Error,
}
